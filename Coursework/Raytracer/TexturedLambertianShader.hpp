#pragma once
#include "Shader.hpp"
#include <cmath>

/// <summary>
/// Lambertian reflectance shader that samples albedo values from a texture.
/// The texture should be stored as an image file (TGAImage instance).
/// </summary>
class TexturedLambertianShader : public Shader
{
private:
	const std::vector<uint8_t>* albedoTexture_;
	const int texWidth_, texHeight_;
	bool shadowTest_;
	float tileU_ =1.f;
	float tileV_ =1.f;

	static float wrap01(float v)
	{
		// Repeat addressing: wrap any value (including negatives) into [0,1).
		v = v - std::floor(v);
		// Handle the rare case where v becomes1 due to FP error.
		if (v >=1.f) v =0.f;
		return v;
	}
public:
	TexturedLambertianShader(const std::vector<uint8_t>* albedoTexture, int texWidth, int texHeight, bool shadowTest=true)
		:shadowTest_(shadowTest), albedoTexture_(albedoTexture),
		texWidth_(texWidth), texHeight_(texHeight)
	{}

	// Optional tiling: set tileU/tileV >1 to repeat the texture more times.
	TexturedLambertianShader(const std::vector<uint8_t>* albedoTexture, int texWidth, int texHeight, float tileU, float tileV, bool shadowTest=true)
		:shadowTest_(shadowTest), albedoTexture_(albedoTexture),
		texWidth_(texWidth), texHeight_(texHeight), tileU_(tileU), tileV_(tileV)
	{}

	virtual Eigen::Vector3f getColor(const HitInfo& hitInfo, 
		const Renderable* scene, 
		const std::vector<std::unique_ptr<Light>>& lights,
		const Eigen::Vector3f& ambientLight,
		int currBounceCount,
		const int maxBounces) const
	{
		Eigen::Vector3f albedo;

		Eigen::Vector2f tex = hitInfo.texCoords;
		float u = wrap01(tex.x() * tileU_);
		float v = wrap01(tex.y() * tileV_);

		// Convert to pixel coordinates. Use (1 - v) to flip V for image space.
		int pixX = static_cast<int>(u * static_cast<float>(texWidth_));
		int pixY = static_cast<int>((1.f - v) * static_cast<float>(texHeight_));

		// Clamp to valid pixel indices (in case u/v lands exactly on1.0).
		pixX = std::min(std::max(pixX,0), texWidth_ -1);
		pixY = std::min(std::max(pixY,0), texHeight_ -1);

		albedo.x() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_ * pixY) *4 +0]) /255.f;
		albedo.y() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_ * pixY) *4 +1]) /255.f;
		albedo.z() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_ * pixY) *4 +2]) /255.f;

		Eigen::Vector3f color = coefftWiseMul(albedo, ambientLight);

		for (auto& light : lights) {
			if (shadowTest_) {
				if (!light->visibilityCheck(hitInfo.location, scene))
					continue;
			}
			Eigen::Vector3f lightVec = light->getVecToLight(hitInfo.location);
			float dotProd = std::max(lightVec.dot(hitInfo.normal),0.f);
			color += dotProd * coefftWiseMul(light->getIntensity(hitInfo.location), albedo);
		}

		return color;
	}
};

