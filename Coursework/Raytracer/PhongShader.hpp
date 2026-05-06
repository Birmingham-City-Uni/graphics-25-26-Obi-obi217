#pragma once
#include "Shader.hpp"
#include <cmath>
#include <algorithm>

/// <summary>
/// Shader using the classic Phong reflectance model to add specular highlights.
/// Albedo (diffuse) is always sampled from a texture (like `TexturedLambertianShader`).
/// </summary>
class PhongShader : public Shader
{
private:
	const std::vector<uint8_t>* albedoTexture_;
	int texWidth_;
	int texHeight_;
	float tileU_ =1.f;
	float tileV_ =1.f;

	Eigen::Vector3f specular_;
	float shininess_;
	bool shadowTest_;

	static float wrap01(float v)
	{
		v = v - std::floor(v);
		if (v >=1.f) v =0.f;
		return v;
	}

	Eigen::Vector3f sampleAlbedo(const HitInfo& hitInfo) const
	{
		Eigen::Vector2f tex = hitInfo.texCoords;
		float u = wrap01(tex.x() * tileU_);
		float v = wrap01(tex.y() * tileV_);

		int pixX = static_cast<int>(u * static_cast<float>(texWidth_));
		int pixY = static_cast<int>((1.f - v) * static_cast<float>(texHeight_));

		pixX = std::min(std::max(pixX,0), texWidth_ -1);
		pixY = std::min(std::max(pixY,0), texHeight_ -1);

		Eigen::Vector3f albedo;
		albedo.x() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_ * pixY) *4 +0]) /255.f;
		albedo.y() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_ * pixY) *4 +1]) /255.f;
		albedo.z() = static_cast<float>((*albedoTexture_)[(pixX + texWidth_ * pixY) *4 +2]) /255.f;
		return albedo;
	}

public:
	// Textured albedo Phong (no tiling)
	PhongShader(const std::vector<uint8_t>* albedoTexture, int texWidth, int texHeight,
		const Eigen::Vector3f& specular, float shininess, bool shadowTest = true)
		: albedoTexture_(albedoTexture)
		, texWidth_(texWidth)
		, texHeight_(texHeight)
		, specular_(specular)
		, shininess_(shininess)
		, shadowTest_(shadowTest)
	{
	}

	// Textured albedo Phong with tiling
	PhongShader(const std::vector<uint8_t>* albedoTexture, int texWidth, int texHeight,
		float tileU, float tileV,
		const Eigen::Vector3f& specular, float shininess, bool shadowTest = true)
		: albedoTexture_(albedoTexture)
		, texWidth_(texWidth)
		, texHeight_(texHeight)
		, tileU_(tileU)
		, tileV_(tileV)
		, specular_(specular)
		, shininess_(shininess)
		, shadowTest_(shadowTest)
	{
	}

	virtual Eigen::Vector3f getColor(const HitInfo& hitInfo,
		const Renderable* scene,
		const std::vector<std::unique_ptr<Light>>& lights,
		const Eigen::Vector3f& ambientLight,
		int currBounceCount,
		const int maxBounces) const override
	{
		Eigen::Vector3f albedo = sampleAlbedo(hitInfo);
		Eigen::Vector3f color = coefftWiseMul(albedo, ambientLight);

		for (auto& light : lights) {
			if (shadowTest_) {
				if (!light->visibilityCheck(hitInfo.location, scene))
					continue;
			}
			Eigen::Vector3f lightVec = light->getVecToLight(hitInfo.location);
			float dotProd = std::max(lightVec.dot(hitInfo.normal),0.f);
			color += dotProd * coefftWiseMul(light->getIntensity(hitInfo.location), albedo);

			Eigen::Vector3f reflectVec = reflect(hitInfo.inDirection, hitInfo.normal);
			float dotSpec = std::max(lightVec.dot(reflectVec),0.f);
			dotSpec = std::pow(dotSpec, shininess_);
			color += dotSpec * coefftWiseMul(light->getIntensity(hitInfo.location), specular_);
		}

		return color;
	}
};

