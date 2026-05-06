#pragma once
#include "Shader.hpp"
#include "GeomUtil.hpp"

/// <summary>
/// Shader modelling perfect mirror reflectance.
/// </summary>
class GlassShader : public Shader	
{
public:
	float _reIndex;
	Eigen::Vector3f _albedo;
	int _opacity;

	GlassShader(float reIndex, Eigen::Vector3f albedo, int opacity) 
	{
		_reIndex = reIndex;
		_albedo = albedo;
		_opacity = opacity;
	}

	virtual Eigen::Vector3f getColor(const HitInfo& hitInfo,
		const Renderable* scene,
		const std::vector<std::unique_ptr<Light>>& lights,
		const Eigen::Vector3f& ambientLight,
		int currBounceCount,
		const int maxBounces) const
	{
		if (currBounceCount >= maxBounces) return Eigen::Vector3f::Zero();

		Ray refractionRay;
		refractionRay.direction = refract(hitInfo.inDirection.normalized(), hitInfo.normal.normalized(), _reIndex);
		refractionRay.origin = hitInfo.location;

		Eigen::Vector3f color = Eigen::Vector3f::Zero();

		HitInfo refractionHit;
		if (scene->intersect(refractionRay, 1e-6f, 1e4f, refractionHit, VISIBLE_BITMASK)) {
			color = refractionHit.shader->getColor(
				refractionHit, scene,
				lights, ambientLight,
				currBounceCount + 1, maxBounces);
		}

		if (_opacity < 255) {
			float alpha = _opacity / 255.0f;
			color.x() = alpha * color.x() + (1 - alpha) * _albedo.x();
			color.y() = alpha * color.y() + (1 - alpha) * _albedo.y();
			color.z() = alpha * color.z() + (1 - alpha) * _albedo.z();
		}

		return color;
	}
};
