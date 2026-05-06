// This define is necessary to get the M_PI constant.
#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <lodepng.h>
#include "Image.hpp"
#include "LinAlg.hpp"
#include "Light.hpp"
#include "Mesh.hpp"
#include "Shading.hpp"

struct Triangle {
	std::array<Eigen::Vector3f, 3> screen; // Coordinates of the triangle in screen space.
	std::array<Eigen::Vector3f, 3> verts; // Vertices of the triangle in world space.
	std::array<Eigen::Vector3f, 3> norms; // Normals of the triangle corners in world space.
	std::array<Eigen::Vector2f, 3> texs; // Texture coordinates of the triangle corners.
};

struct Texture
{
	unsigned width = 0;
	unsigned height = 0;
	std::vector<unsigned char> rgba; // size = width*height*4
};

static Texture loadTexturePNG(const std::string& filename)
{
	Texture t;
	unsigned error = lodepng::decode(t.rgba, t.width, t.height, filename);
	if (error) throw std::runtime_error(std::string("lodepng error: ") + lodepng_error_text(error));
	return t;
}

static inline float clamp01(float v)
{
	return std::max(0.0f, std::min(1.0f, v));
}

static inline uint8_t u8Clamp(int v)
{
	return (uint8_t)std::max(0, std::min(255, v));
}

static inline size_t pxIndex(int x, int y, int width)
{
	return 4ull * (size_t(x) + size_t(y) * size_t(width));
}

// Simple depth-of-field post-process using zBuffer in NDC space (0..1-ish in your projection).
// focusDepth: depth value that stays sharp (try ~0.94-0.99 depending on your scene)
// focusRange: how much depth around focusDepth remains mostly sharp
// maxRadius: maximum blur radius in pixels
static void applyDepthOfField(
	std::vector<uint8_t>& rgba,
	int width, int height,
	const std::vector<float>& zBuffer,
	float focus01,       // focus in normalized depth [0..1]
	float focusRange01,  // in-focus range in normalized depth
	int maxRadius)
{
	// Find min/max depth actually written (ignore background = 1.0)
	float zMin = std::numeric_limits<float>::infinity();
	float zMax = -std::numeric_limits<float>::infinity();

	for (float z : zBuffer)
	{
		if (z >= 0.9999f) continue;
		zMin = std::min(zMin, z);
		zMax = std::max(zMax, z);
	}

	if (!std::isfinite(zMin) || !std::isfinite(zMax) || (zMax - zMin) < 1e-6f)
		return;

	std::vector<uint8_t> src = rgba; // read from src, write to rgba

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			const int idxDepth = x + y * width;
			const float z = zBuffer[idxDepth];

			if (z >= 0.9999f)
				continue;

			// Normalize depth into [0..1] across what you actually rendered
			const float z01 = (z - zMin) / (zMax - zMin);

			// Blur only far pixels:
			// focus01 = farStart01, focusRange01 = farEnd01 (repurposed names)
			const float farStart01 = focus01;
			const float farEnd01 = std::max(farStart01 + 1e-6f, focusRange01);

			// 0 before farStart01, 1 at/after farEnd01
			float coc = (z01 - farStart01) / (farEnd01 - farStart01);
			coc = clamp01(coc);

			// optional: smoother ramp (comment out if you want linear)
			coc = coc * coc * (3.0f - 2.0f * coc);

			const int r = (int)std::lround(coc * maxRadius);
			if (r <= 0)
				continue;

			int sumR = 0, sumG = 0, sumB = 0, sumA = 0;
			int count = 0;

			const int x0 = std::max(0, x - r);
			const int x1 = std::min(width - 1, x + r);
			const int y0 = std::max(0, y - r);
			const int y1 = std::min(height - 1, y + r);

			for (int yy = y0; yy <= y1; ++yy)
			{
				for (int xx = x0; xx <= x1; ++xx)
				{
					const size_t pi = pxIndex(xx, yy, width);
					sumR += src[pi + 0];
					sumG += src[pi + 1];
					sumB += src[pi + 2];
					sumA += src[pi + 3];
					++count;
				}
			}

			const size_t po = pxIndex(x, y, width);
			rgba[po + 0] = u8Clamp(sumR / count);
			rgba[po + 1] = u8Clamp(sumG / count);
			rgba[po + 2] = u8Clamp(sumB / count);
			rgba[po + 3] = u8Clamp(sumA / count);
		}
	}
}

static float wrap01(float x)
{
	// wraps any float into [0,1)
	x = x - floorf(x);
	if (x < 0.0f) x += 1.0f;
	return x;
}

static Eigen::Vector3f sampleTexture(const Texture& tex, const Eigen::Vector2f& uv)
{
	// WRAP UVs instead of clamping (supports tiling, avoids "stretched" look)
	float u = wrap01(uv.x());
	float v = wrap01(uv.y());

	// OBJ UVs usually have v=0 at bottom; images often have y=0 at top -> flip v
	v = 1.0f - v;

	int x = (int)std::round(u * (tex.width - 1));
	int y = (int)std::round(v * (tex.height - 1));

	const size_t idx = 4ull * (size_t(x) + size_t(y) * tex.width);

	float r = tex.rgba[idx + 0] / 255.0f;
	float g = tex.rgba[idx + 1] / 255.0f;
	float b = tex.rgba[idx + 2] / 255.0f;

	return Eigen::Vector3f(r, g, b);
}

static Texture makeSolidTexture(const Eigen::Vector3f& rgb01)
{
	Texture t;
	t.width = 1;
	t.height = 1;
	t.rgba.resize(4);

	const float r = std::clamp(rgb01.x(), 0.0f, 1.0f);
	const float g = std::clamp(rgb01.y(), 0.0f, 1.0f);
	const float b = std::clamp(rgb01.z(), 0.0f, 1.0f);

	t.rgba[0] = (unsigned char)std::lround(r * 255.0f);
	t.rgba[1] = (unsigned char)std::lround(g * 255.0f);
	t.rgba[2] = (unsigned char)std::lround(b * 255.0f);
	t.rgba[3] = 255;
	return t;
}

Eigen::Matrix4f projectionMatrix(int height, int width, float horzFov = 35.f * M_PI / 180.f, float zFar = 220.f, float zNear = 0.1f)
{
	float vertFov = horzFov * float(height) / width;
	Eigen::Matrix4f projection;
	projection <<
		1.0f / tanf(0.5f * horzFov), 0, 0, 0,
		0, 1.0f / tanf(0.5f * vertFov), 0, 0,
		0, 0, zFar / (zFar - zNear), -zFar * zNear / (zFar - zNear),
		0, 0, 1, 0;
	return projection;
}

void findScreenBoundingBox(const Triangle& t, int width, int height, int& minX, int& minY, int& maxX, int& maxY)
{
	// Find a bounding box around the triangle
	minX = std::min(std::min(t.screen[0].x(), t.screen[1].x()), t.screen[2].x());
	minY = std::min(std::min(t.screen[0].y(), t.screen[1].y()), t.screen[2].y());
	maxX = std::max(std::max(t.screen[0].x(), t.screen[1].x()), t.screen[2].x());
	maxY = std::max(std::max(t.screen[0].y(), t.screen[1].y()), t.screen[2].y());

	// Constrain it to lie within the image.
	minX = std::min(std::max(minX, 0), width - 1);
	maxX = std::min(std::max(maxX, 0), width - 1);
	minY = std::min(std::max(minY, 0), height - 1);
	maxY = std::min(std::max(maxY, 0), height - 1);
}


void drawTriangle(std::vector<uint8_t>& image, int width, int height,
	std::vector<float>& zBuffer,
	const Triangle& t,
	const std::vector<std::unique_ptr<Light>>& lights,
	const Texture& tex, const Eigen::Vector3f& specularColor,
	float specularExponent,
	const Eigen::Vector3f& camWorldPos)
{
	int minX, minY, maxX, maxY;
	findScreenBoundingBox(t, width, height, minX, minY, maxX, maxY);

	Eigen::Vector2f edge1 = v2(t.screen[2] - t.screen[0]);
	Eigen::Vector2f edge2 = v2(t.screen[1] - t.screen[0]);
	float triangleArea = 0.5f * vec2Cross(edge2, edge1);
	if (triangleArea < 0) {
		// Triangle is backfacing
		// Exit and quit drawing!
		return;
	}

	for (int x = minX; x <= maxX; ++x)
		for (int y = minY; y <= maxY; ++y) {
			Eigen::Vector2f p(x, y);

			// Find sub-triangle areas
			float a0 = 0.5f * fabsf(vec2Cross(v2(t.screen[1]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a1 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a2 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[1]), p - v2(t.screen[1])));

			// find barycentrics
			float b0 = a0 / triangleArea;
			float b1 = a1 / triangleArea;
			float b2 = a2 / triangleArea;

			// If outside triangle, exit early
			float sum = b0 + b1 + b2;
			if (sum > 1.0001) {
				continue;
			}

			Eigen::Vector3f worldP = t.verts[0] * b0 + t.verts[1] * b1 + t.verts[2] * b2;

			float depth = t.screen[0].z() * b0 + t.screen[1].z() * b1 + t.screen[2].z() * b2;
			int depthIdx = static_cast<int>(p.x()) + static_cast<int>(p.y()) * width;
			if (depth > zBuffer[depthIdx]) continue;
			zBuffer[depthIdx] = depth;

			Eigen::Vector3f normP = t.norms[0] * b0 + t.norms[1] * b1 + t.norms[2] * b2;
			normP.normalize();

			// Work out colour at this position.
			Eigen::Vector3f color = Eigen::Vector3f::Zero();
			Eigen::Vector2f uvP = t.texs[0] * b0 + t.texs[1] * b1 + t.texs[2] * b2;
			Eigen::Vector3f albedo = sampleTexture(tex, uvP);				

			// Convert sRGB (PNG) -> linear for shading
			albedo = Eigen::Vector3f(
				powf(albedo.x(), 2.2f),
				powf(albedo.y(), 2.2f),
				powf(albedo.z(), 2.2f)
			);

			// Iterate over lights, and sum to find colour.
			for (auto& light : lights) {

				// Work out the contribution from this light source, and add it to the color variable.

				// Work out the intensity of this light source, at the point worldP.
				Eigen::Vector3f lightIntensity = light->getIntensityAt(worldP);

				// We only need to do the following if the light isn't an ambient light.
				if (light->getType() != Light::Type::AMBIENT) {

					// Subtask 3: Work out correct inputs for the phongSpecularTerm function inside drawTriangle, and draw an image!
					// *** YOUR CODE HERE ***
					// Work out the incoming light dir (from the light into the surface point).
					Eigen::Vector3f incomingLightDir = light->getDirection(worldP);
					// Work out the view direction (from surface point towards camera). Make sure it's normalized!
					Eigen::Vector3f viewDir = (camWorldPos - worldP).normalized();
					// Find the specular term by calling phongSpecularTerm.
					float specularTerm = phongSpecularTerm(incomingLightDir, normP, viewDir, specularExponent);
					// *** END YOUR CODE ***

					Eigen::Vector3f specularOut = specularColor * specularTerm;
					specularOut = coeffWiseMultiply(specularOut, lightIntensity);

					// Take the dot product of the normal with the light direction.
					float dotProd = normP.dot(-incomingLightDir);

					// We don't want negative light - if dot product less than 0, set it to 0.
					dotProd = std::max(dotProd, 0.0f);

					// Multiply the light intensity by the dot product.
					Eigen::Vector3f diffuseOut = lightIntensity * dotProd;
					diffuseOut = coeffWiseMultiply(diffuseOut, albedo);

					// Add both diffuse and specular components to the colour.
					color += specularOut;
					color += diffuseOut;
				}
				else {
					// Light is ambient - just multiply light intensity with albedo.
					color += coeffWiseMultiply(lightIntensity, albedo);
				}
			}

			// Clamp in linear space before gamma-encoding (prevents NaNs and weird desaturation)
			color = color.cwiseMax(0.0f);
			color = color.cwiseMin(1.0f);

			Color c;
			// Gamma-correcting colours.
			c.r = (uint8_t)(std::min(powf(color.x(), 1.0f / 2.2f), 1.0f) * 255.0f);
			c.g = (uint8_t)(std::min(powf(color.y(), 1.0f / 2.2f), 1.0f) * 255.0f);
			c.b = (uint8_t)(std::min(powf(color.z(), 1.0f / 2.2f), 1.0f) * 255.0f);
			c.a = 255;

			setPixel(image, x, y, width, height, c);
		}
}

void drawMeshPart(std::vector<unsigned char>& image,
	std::vector<float>& zBuffer,
	const Mesh& mesh,
	int faceStart, int faceEnd,              // NEW: which faces to draw
	const Texture& tex, const Eigen::Vector3f& specularColor,
	float specularExponent,
	const Eigen::Vector3f& camWorldPos,
	const Eigen::Matrix4f& modelToWorld,
	const Eigen::Matrix4f& worldToClip,
	const std::vector<std::unique_ptr<Light>>& lights,
	int width, int height)
{
	for (int i = faceStart; i < faceEnd; ++i) {
		Eigen::Vector3f
			v0 = mesh.verts[mesh.vFaces[i][0]],
			v1 = mesh.verts[mesh.vFaces[i][1]],
			v2 = mesh.verts[mesh.vFaces[i][2]];
		Eigen::Vector3f
			n0 = mesh.norms[mesh.nFaces[i][0]],
			n1 = mesh.norms[mesh.nFaces[i][1]],
			n2 = mesh.norms[mesh.nFaces[i][2]];

		Triangle t;
		t.verts[0] = (modelToWorld * vec3ToVec4(v0)).block<3, 1>(0, 0);
		t.verts[1] = (modelToWorld * vec3ToVec4(v1)).block<3, 1>(0, 0);
		t.verts[2] = (modelToWorld * vec3ToVec4(v2)).block<3, 1>(0, 0);

		Eigen::Vector4f vClip0 = worldToClip * modelToWorld * vec3ToVec4(v0);
		vClip0 /= vClip0.w();
		Eigen::Vector4f vClip1 = worldToClip * modelToWorld * vec3ToVec4(v1);
		vClip1 /= vClip1.w();
		Eigen::Vector4f vClip2 = worldToClip * modelToWorld * vec3ToVec4(v2);
		vClip2 /= vClip2.w();

		if (outsideClipBox(vClip0) || outsideClipBox(vClip1) || outsideClipBox(vClip2)) continue;

		t.screen[0] = Eigen::Vector3f((vClip0.x() + 1.0f) * width / 2, (-vClip0.y() + 1.0f) * height / 2, vClip0.z());
		t.screen[1] = Eigen::Vector3f((vClip1.x() + 1.0f) * width / 2, (-vClip1.y() + 1.0f) * height / 2, vClip1.z());
		t.screen[2] = Eigen::Vector3f((vClip2.x() + 1.0f) * width / 2, (-vClip2.y() + 1.0f) * height / 2, vClip2.z());

		t.norms[0] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n0).normalized();
		t.norms[1] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n1).normalized();
		t.norms[2] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n2).normalized();

		t.texs[0] = mesh.texs[mesh.tFaces[i][0]];
		t.texs[1] = mesh.texs[mesh.tFaces[i][1]];
		t.texs[2] = mesh.texs[mesh.tFaces[i][2]];

		drawTriangle(image, width, height, zBuffer, t, lights, tex, specularColor, specularExponent, camWorldPos);
	}
}

void drawTriangle(std::vector<uint8_t>& image, int width, int height,
	std::vector<float>& zBuffer,
	const Triangle& t,
	const std::vector<std::unique_ptr<Light>>& lights,
	const Eigen::Vector3f& albedo, float opacity, const Eigen::Vector3f& specularColor,
	float specularExponent,
	const Eigen::Vector3f& camWorldPos)
{
	int minX, minY, maxX, maxY;
	findScreenBoundingBox(t, width, height, minX, minY, maxX, maxY);

	Eigen::Vector2f edge1 = v2(t.screen[2] - t.screen[0]);
	Eigen::Vector2f edge2 = v2(t.screen[1] - t.screen[0]);
	float triangleArea = 0.5f * vec2Cross(edge2, edge1);
	if (triangleArea < 0) {
		// Triangle is backfacing
		// Exit and quit drawing!
		return;
	}

	for (int x = minX; x <= maxX; ++x) {
		for (int y = minY; y <= maxY; ++y) {
			Eigen::Vector2f p(x, y);

			// Find sub-triangle areas
			float a0 = 0.5f * fabsf(vec2Cross(v2(t.screen[1]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a1 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a2 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[1]), p - v2(t.screen[1])));

			// find barycentrics
			float b0 = a0 / triangleArea;
			float b1 = a1 / triangleArea;
			float b2 = a2 / triangleArea;

			// If outside triangle, exit early
			float sum = b0 + b1 + b2;
			if (sum > 1.0001) {
				continue;
			}

			Eigen::Vector3f worldP = t.verts[0] * b0 + t.verts[1] * b1 + t.verts[2] * b2;

			float depth = t.screen[0].z() * b0 + t.screen[1].z() * b1 + t.screen[2].z() * b2;
			int depthIdx = static_cast<int>(p.x()) + static_cast<int>(p.y()) * width;
			if (depth > zBuffer[depthIdx]) continue;
			zBuffer[depthIdx] = depth;

			Eigen::Vector3f normP = t.norms[0] * b0 + t.norms[1] * b1 + t.norms[2] * b2;
			normP.normalize();

			// Work out colour at this position.
			Eigen::Vector3f color = Eigen::Vector3f::Zero();
			Eigen::Vector2f uvP = t.texs[0] * b0 + t.texs[1] * b1 + t.texs[2] * b2;
			//::Vector3f albedo = sampleTexture(tex, uvP);

			// Convert sRGB (PNG) -> linear for shading
			Eigen::Vector3f albedoLin{
				powf(albedo.x(), 2.2f),
				powf(albedo.y(), 2.2f),
				powf(albedo.z(), 2.2f)
			};

			// Iterate over lights, and sum to find colour.
			for (auto& light : lights) {

				// Work out the contribution from this light source, and add it to the color variable.

				// Work out the intensity of this light source, at the point worldP.
				Eigen::Vector3f lightIntensity = light->getIntensityAt(worldP);

				// We only need to do the following if the light isn't an ambient light.
				if (light->getType() != Light::Type::AMBIENT) {

					// Subtask 3: Work out correct inputs for the phongSpecularTerm function inside drawTriangle, and draw an image!
					// *** YOUR CODE HERE ***
					// Work out the incoming light dir (from the light into the surface point).
					Eigen::Vector3f incomingLightDir = light->getDirection(worldP);
					// Work out the view direction (from surface point towards camera). Make sure it's normalized!
					Eigen::Vector3f viewDir = (camWorldPos - worldP).normalized();
					// Find the specular term by calling phongSpecularTerm.
					float specularTerm = phongSpecularTerm(incomingLightDir, normP, viewDir, specularExponent);
					// *** END YOUR CODE ***

					Eigen::Vector3f specularOut = specularColor * specularTerm;
					specularOut = coeffWiseMultiply(specularOut, lightIntensity);

					// Take the dot product of the normal with the light direction.
					float dotProd = normP.dot(-incomingLightDir);

					// We don't want negative light - if dot product less than 0, set it to 0.
					dotProd = std::max(dotProd, 0.0f);

					// Multiply the light intensity by the dot product.
					Eigen::Vector3f diffuseOut = lightIntensity * dotProd;
					diffuseOut = coeffWiseMultiply(diffuseOut, albedoLin);

					// Add both diffuse and specular components to the colour.
					color += specularOut;
					color += diffuseOut;
				}
				else {
					// Light is ambient - just multiply light intensity with albedo.
					color += coeffWiseMultiply(lightIntensity, albedoLin);
				}
			}

			// Clamp in linear space before gamma-encoding (prevents NaNs and weird desaturation)
			color = color.cwiseMax(0.0f);
			color = color.cwiseMin(1.0f);

			Color c;
			// Gamma-correcting colours.
			c.r = (uint8_t)(std::min(powf(color.x(), 1.0f / 2.2f), 1.0f) * 255.0f);
			c.g = (uint8_t)(std::min(powf(color.y(), 1.0f / 2.2f), 1.0f) * 255.0f);
			c.b = (uint8_t)(std::min(powf(color.z(), 1.0f / 2.2f), 1.0f) * 255.0f);
			c.a = 255;

			if (opacity < 255) {
				float alpha = opacity / 255.0f;

				c.r = (uint8_t)(c.r * alpha);
				c.g = (uint8_t)(c.g * alpha);
				c.b = (uint8_t)(c.b * alpha);

				Color cO = getPixel(image, x, y, width, height);

				cO.r = (uint8_t)(cO.r * (1.0f - alpha));
				cO.g = (uint8_t)(cO.g * (1.0f - alpha));
				cO.b = (uint8_t)(cO.b * (1.0f - alpha));

				Color cOut{ 0, 0, 0, 255 };

				cOut.r = std::min(255, c.r + cO.r);
				cOut.g = std::min(255, c.g + cO.g);
				cOut.b = std::min(255, c.b + cO.b);

				setPixel(image, x, y, width, height, cOut);
			}

			else {
				setPixel(image, x, y, width, height, c);
			}
		}
	}
}

void drawMeshPart(std::vector<unsigned char>& image,
	std::vector<float>& zBuffer,
	const Mesh& mesh,
	int faceStart, int faceEnd,              // NEW: which faces to draw
	const Eigen::Vector3f& albedo, float opacity, const Eigen::Vector3f& specularColor,
	float specularExponent,
	const Eigen::Vector3f& camWorldPos,
	const Eigen::Matrix4f& modelToWorld,
	const Eigen::Matrix4f& worldToClip,
	const std::vector<std::unique_ptr<Light>>& lights,
	int width, int height)
{
	for (int i = faceStart; i < faceEnd; ++i) {
		Eigen::Vector3f
			v0 = mesh.verts[mesh.vFaces[i][0]],
			v1 = mesh.verts[mesh.vFaces[i][1]],
			v2 = mesh.verts[mesh.vFaces[i][2]];
		Eigen::Vector3f
			n0 = mesh.norms[mesh.nFaces[i][0]],
			n1 = mesh.norms[mesh.nFaces[i][1]],
			n2 = mesh.norms[mesh.nFaces[i][2]];

		Triangle t;
		t.verts[0] = (modelToWorld * vec3ToVec4(v0)).block<3, 1>(0, 0);
		t.verts[1] = (modelToWorld * vec3ToVec4(v1)).block<3, 1>(0, 0);
		t.verts[2] = (modelToWorld * vec3ToVec4(v2)).block<3, 1>(0, 0);

		Eigen::Vector4f vClip0 = worldToClip * modelToWorld * vec3ToVec4(v0);
		vClip0 /= vClip0.w();
		Eigen::Vector4f vClip1 = worldToClip * modelToWorld * vec3ToVec4(v1);
		vClip1 /= vClip1.w();
		Eigen::Vector4f vClip2 = worldToClip * modelToWorld * vec3ToVec4(v2);
		vClip2 /= vClip2.w();

		if (outsideClipBox(vClip0) || outsideClipBox(vClip1) || outsideClipBox(vClip2)) continue;

		t.screen[0] = Eigen::Vector3f((vClip0.x() + 1.0f) * width / 2, (-vClip0.y() + 1.0f) * height / 2, vClip0.z());
		t.screen[1] = Eigen::Vector3f((vClip1.x() + 1.0f) * width / 2, (-vClip1.y() + 1.0f) * height / 2, vClip1.z());
		t.screen[2] = Eigen::Vector3f((vClip2.x() + 1.0f) * width / 2, (-vClip2.y() + 1.0f) * height / 2, vClip2.z());

		t.norms[0] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n0).normalized();
		t.norms[1] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n1).normalized();
		t.norms[2] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n2).normalized();

		t.texs[0] = mesh.texs[mesh.tFaces[i][0]];
		t.texs[1] = mesh.texs[mesh.tFaces[i][1]];
		t.texs[2] = mesh.texs[mesh.tFaces[i][2]];

		drawTriangle(image, width, height, zBuffer, t, lights, albedo, opacity, specularColor, specularExponent, camWorldPos);
	}
}

int main()
{
	std::string outputFilename = "output.png";

	const int width = 1920, height = 1080;
	const int nChannels = 4;

	// Setting up an image buffer
	// This std::vector has one 8-bit value for each pixel in each row and column of the image, and
	// for each of the 4 channels (red, green, blue and alpha).
	// Remember 8-bit unsigned values can range from 0 to 255.
	std::vector<uint8_t> imageBuffer(height * width * nChannels);
	std::vector<float> zBuffer(height * width);

	// This line sets the image to black initially.
	Color black{ 0,0,0,255 };
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			setPixel(imageBuffer, c, r, width, height, black);
			zBuffer[r * width + c] = 1.0f;
		}
	}

	Eigen::Matrix4f projection = projectionMatrix(height, width);

	// This matrix rotates the camera, tilting it down, then translates it up to make it look down on the scene.
	Eigen::Matrix4f cameraToWorld = translationMatrix(Eigen::Vector3f(0.8f, 1.f, 1.1f)) * rotateYMatrix(-0.13f) * rotateXMatrix(0.2f);

	Eigen::Vector3f camWorldPos = (cameraToWorld * Eigen::Vector4f(0, 0, 0, 1)).block<3, 1>(0, 0);

	// The main important task = set up the worldToCamera and worldToClip matrices here!
	// Set up worldToCamera, based on cameraToWorld above
	Eigen::Matrix4f worldToCamera = cameraToWorld.inverse();
	// Set up worldToClip, using the projection and worldToCamera matrices
	Eigen::Matrix4f worldToClip = projection * worldToCamera;

	// *** END YOUR CODE ***

	std::string entireSceneFilename = "../models/entire_scene.obj";

	// Subtask 4: Try re-rendering your image with different lighting setups, and specular exponents, and see how it changes!
	// You can modify the lighting setup here....
	std::vector<std::unique_ptr<Light>> lights;
	lights.emplace_back(new AmbientLight(Eigen::Vector3f(0.1f, 0.1f, 0.1f)));
	lights.emplace_back(new DirectionalLight(Eigen::Vector3f(0.4f, 0.4f, 0.4f), Eigen::Vector3f(1.f, -1.f, 0.0f)));

	Mesh entireSceneMesh = loadMeshFile(entireSceneFilename);

	// load textures (replace filenames with yours)
	Texture texDefault = makeSolidTexture(Eigen::Vector3f(1.0f, 0.0f, 1.0f)); // magenta fallback
	Texture texTrash = loadTexturePNG("../models/blinn1_Base_color.png");
	Texture texPlane = loadTexturePNG("../models/coast_sand_01_diff_2k.png");
	Texture texEye = loadTexturePNG("../models/Eye_BaseColor_tga.png");
	Texture texMask = loadTexturePNG("../models/Char_cyber_Mask_BaseColor.png");
	Texture texHead = loadTexturePNG("../models/Char_cyber_Head_BaseColor.png");
	Texture texChest = loadTexturePNG("../models/Char_cyber_Chest1_BaseColor.png");
	Texture texLeg = loadTexturePNG("../models/Char_cyber_Legs1_BaseColor.png");
	Texture texOG = loadTexturePNG("../models/Buick_GNX_1987_Grille2A_DiffuseAOSO.png");
	Texture texOne = loadTexturePNG("../models/One.png");
	Texture texTwo = loadTexturePNG("../models/Two.png");
	Texture texThree = loadTexturePNG("../models/Buick_GNX_1987_ManufacturerPlateA_Diffuse.png");
	Texture texFour = loadTexturePNG("../models/Buick_GNX_1987_Grille5A_DiffuseAOSO.png");
	Texture texFive = loadTexturePNG("../models/Buick_GNX_1987_InteriorA_Emissive.png");
	Texture texSix = loadTexturePNG("../models/Global_Texture_Coloured_Diffuse_P.png");
	Texture texSeven = loadTexturePNG("../models/Buick_GNX_1987_Grille3A_DiffuseAOSO.png");
	Texture texEight = loadTexturePNG("../models/Buick_GNX_1987_Grille4A_DiffuseAOSO.png");
	Texture texNine = loadTexturePNG("../models/Buick_GNX_1987_TexturedA_Diffuse.png");
	Texture texTen = loadTexturePNG("../models/BadgeA_DiffuseAOSO.png");
	Texture texEleven = loadTexturePNG("../models/One.png");
	Texture texTwelve = loadTexturePNG("../models/One.png");
	Texture texThirteen = loadTexturePNG("../models/Thirteen.png");
	Texture texFourteen = loadTexturePNG("../models/Fourteen.png");
	Texture texFifteen = loadTexturePNG("../models/EngineA_DiffuseAOSO.png");
	Texture texSixteen = loadTexturePNG("../models/Buick_GNX_1987_Textured2A_Diffuse.png");
	Texture texSeventeen = loadTexturePNG("../models/Buick_GNX_1987_InteriorTillingA_DiffuseAOSO.png");
	Texture texEighteen = loadTexturePNG("../models/Buick_GNX_1987_LightA_Emissive.png");
	Texture texNineteen = loadTexturePNG("../models/Buick_GNX_1987_Grille1A_DiffuseAOSO.png");
	Texture texTwenty = loadTexturePNG("../models/InteriorA_DiffuseAOSO.png");
	Texture texTwentyOne = loadTexturePNG("../models/Buick_GNX_1987_LightA_Diffuse.png");
	Texture texTwentyTwo = loadTexturePNG("../models/Twentytwo.png");
	Texture texTwentyThree = loadTexturePNG("../models/Buick_GNX_1987_Grille6A_DiffuseAOSO.png");
	Texture texWheel = loadTexturePNG("../models/Wheel1A_DiffuseAOSO.png");
	Texture texCalliperZone = loadTexturePNG("../models/Buick_GNX_1987_CalliperA_Zone_Diffuse.png");
	Texture texBackground = loadTexturePNG("../models/Background.png");


	Eigen::Matrix4f entireSceneTransform;
	entireSceneTransform = translationMatrix(Eigen::Vector3f(0.0f, 0.f, 7.f)) * scaleMatrix(0.35f) * rotateYMatrix(M_PI);

	const std::string drawLastName = "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.002";

	// draw each 'o' object separately
	for (int k = 0; k < (int)entireSceneMesh.parts.size(); ++k)
	{
		const std::string& name = entireSceneMesh.parts[k].name;
		if (name == drawLastName) {
			continue;
		}

		int start = entireSceneMesh.parts[k].startFace;
		int end = (k + 1 < (int)entireSceneMesh.parts.size())
			? entireSceneMesh.parts[k + 1].startFace
			: (int)entireSceneMesh.vFaces.size();

		// pick texture for this object name
		const Texture* chosen = &texDefault;
		if (name == "trashpile_3_low.001") chosen = &texTrash;
		else if (name == "Old_trash") chosen = &texTrash;
		else if (name == "Plane") chosen = &texPlane;
		else if (name == "Eye_Inner.Low__Mask.Test.001") chosen = &texEye;
		else if (name == "Eye_Inner.Low__Mask.Test.005") chosen = &texChest;
		else if (name == "Eye_Inner.Low__Mask.Test") chosen = &texMask;
		else if (name == "Eye_Inner.Low__Mask.Test.004") chosen = &texLeg;
		else if (name == "Eye_Inner.Low__Mask.Test.003") chosen = &texHead;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material") chosen = &texOG;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.001") chosen = &texOne;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.003") chosen = &texThree;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.004") chosen = &texFour;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.005") chosen = &texFive;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.006") chosen = &texSix;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.007") chosen = &texSeven;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.008") chosen = &texEight;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.009") chosen = &texNine;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.010") chosen = &texTen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.011") chosen = &texEleven;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.012") chosen = &texTwelve;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.013") chosen = &texThirteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.014") chosen = &texFourteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.015") chosen = &texFifteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.016") chosen = &texSixteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.017") chosen = &texSeventeen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.018") chosen = &texEighteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.019") chosen = &texNineteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.020") chosen = &texTwenty;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.021") chosen = &texTwentyOne;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.022") chosen = &texTwentyTwo;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.023") chosen = &texTwentyThree;
		else if (name == "polySurface1") chosen = &texWheel;
		else if (name == "polySurface331") chosen = &texWheel;
		else if (name == "polySurface661") chosen = &texWheel;
		else if (name == "polySurface979") chosen = &texWheel;
		else if (name == "polySurface980") chosen = &texCalliperZone;
		else if (name == "polySurface1087") chosen = &texCalliperZone;
		else if (name == "polySurface1194") chosen = &texCalliperZone;
		else if (name == "polySurface1400") chosen = &texCalliperZone;
		else if (name == "Plane.001") chosen = &texBackground;

		drawMeshPart(imageBuffer, zBuffer, entireSceneMesh, start, end,
			*chosen, Eigen::Vector3f::Ones() * 1.0f, 10.f, camWorldPos,
			entireSceneTransform, worldToClip, lights, width, height);
		
		
	}

	for (int k = 0; k < (int)entireSceneMesh.parts.size(); ++k)
	{   
		const std::string& name = entireSceneMesh.parts[k].name;
		if (name != drawLastName) {
			continue;
		}

		int start = entireSceneMesh.parts[k].startFace;
		int end = (k + 1 < (int)entireSceneMesh.parts.size())
			? entireSceneMesh.parts[k + 1].startFace
			: (int)entireSceneMesh.vFaces.size();

		drawMeshPart(imageBuffer, zBuffer, entireSceneMesh, start, end,
			Eigen::Vector3f(1, 1, 1), 200.f, Eigen::Vector3f::Ones() * 1.0f, 10.f, camWorldPos,
			entireSceneTransform, worldToClip, lights, width, height);
	}
	// For debug - draw point lights as colored circles so we can see where they are
	drawPointLights(imageBuffer, width, height, lights);
		
	applyDepthOfField(
		imageBuffer, width, height, zBuffer,
		/*farStart01*/ 0.70f,
		/*farEnd01*/   1.00f,
		/*maxRadius*/  13);

	// Save the image to png.
	int errorCode;
	errorCode = lodepng::encode(outputFilename, imageBuffer, width, height);
	if (errorCode) { // check the error code, in case an error occurred.
		std::cout << "lodepng error encoding image: " << lodepng_error_text(errorCode) << std::endl;
		return errorCode;
	}

	saveZBufferImage("zBuffer.png", zBuffer, width, height);

	return 0;
}