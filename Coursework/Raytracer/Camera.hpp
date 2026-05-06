#pragma once
#include "Ray.hpp"

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <algorithm>

/// <summary>
/// Movable camera class. Provide the camera location, forward direction and an up
/// vector, along with the image dimensions and vertical Field of View angle (radians).
/// The camera can then produce a ray passing through each pixel location.
/// 
/// Optional depth-of-field (thin lens):
/// - call `setDepthOfField(aperture, focusDist)` to enable
/// - call `disableDepthOfField()` (or aperture==0) for pinhole behaviour
/// </summary>
class Camera
{
private:
	Eigen::Vector3f location_, bottomLeftPix_, right1pix_, up1pix_;

	// Basis vectors needed for depth of field (world-space).
	Eigen::Vector3f forwardVec_;
	Eigen::Vector3f rightVec_;
	Eigen::Vector3f upVec_;

	// DoF parameters (aperture diameter /2 = radius)
	float lensRadius_ =0.0f;
	float focusDist_ =1.0f;

	static Eigen::Vector2f sampleUnitDisk(std::mt19937& rng)
	{
		std::uniform_real_distribution<float> dist(-1.0f,1.0f);
		for (;;)
		{
			float x = dist(rng);
			float y = dist(rng);
			if (x * x + y * y <=1.0f) return Eigen::Vector2f(x, y);
		}
	}

public:
	Camera(
		const Eigen::Vector3f& location,
		const Eigen::Vector3f& forward,
		const Eigen::Vector3f& up,
		int pixWidth, int pixHeight,
		float vertFov)
		: location_(location)
		, forwardVec_(0.f,0.f,1.f)
		, rightVec_(1.f,0.f,0.f)
		, upVec_(0.f,1.f,0.f)
	{
		forwardVec_ = forward.normalized();
		rightVec_ = (up.cross(forwardVec_)).normalized();
		upVec_ = (forward.cross(rightVec_)).normalized();

		float aspect = static_cast<float>(pixWidth) / static_cast<float>(pixHeight);

		float halfHeight = std::tan(vertFov /2);
		float halfWidth = aspect * halfHeight;

		bottomLeftPix_ = location + forwardVec_ - (halfWidth * rightVec_ + halfHeight * upVec_);

		right1pix_ = rightVec_ * halfWidth *2.f / static_cast<float>(pixWidth);
		up1pix_ = upVec_ * halfHeight *2.f / static_cast<float>(pixHeight);
	}

	void setDepthOfField(float aperture, float focusDist)
	{
		lensRadius_ =0.5f * std::max(0.0f, aperture);
		focusDist_ = (focusDist >0.0f) ? focusDist :1.0f;
	}

	void disableDepthOfField()
	{
		lensRadius_ =0.0f;
	}

	// Original pinhole ray (unchanged)
	Ray getRay(int pixX, int pixY)
	{
		Ray ray;
		ray.origin = location_;
		Eigen::Vector3f pixelPos = bottomLeftPix_ +
			static_cast<float>(pixX) * right1pix_ +
			static_cast<float>(pixY) * up1pix_;

		ray.direction = (pixelPos - location_).normalized();
		return ray;
	}

	// Depth-of-field ray using caller-provided RNG (minimal integration in render loop)
	Ray getRay(int pixX, int pixY, std::mt19937& rng)
	{
		Eigen::Vector3f pixelPos = bottomLeftPix_ +
			static_cast<float>(pixX) * right1pix_ +
			static_cast<float>(pixY) * up1pix_;

		// If DoF disabled, fall back to pinhole behaviour.
		if (lensRadius_ <=0.0f)
			return getRay(pixX, pixY);

		// Focal point along the pinhole ray direction.
		Eigen::Vector3f pinholeDir = (pixelPos - location_).normalized();
		Eigen::Vector3f focalPoint = location_ + pinholeDir * focusDist_;

		Eigen::Vector2f d = sampleUnitDisk(rng) * lensRadius_;
		Eigen::Vector3f lensOffset = rightVec_ * d.x() + upVec_ * d.y();

		Ray ray;
		ray.origin = location_ + lensOffset;
		ray.direction = (focalPoint - ray.origin).normalized();
		return ray;
	}
};

