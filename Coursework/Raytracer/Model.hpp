#pragma once

#include <vector>
#include <Eigen/Dense>
#include <string>

struct VertexIndices
{
	int vert, tex, norm;
};

struct ObjGroup
{
	std::string name;
	int faceStart = 0;
	int faceCount = 0;
};

class Model {
private:
	std::vector<Eigen::Vector3f> verts_, vns_;
	std::vector<Eigen::Vector2f> vts_;
	std::vector<std::vector<VertexIndices>> faces_;

	// NEW: records "o" groups as contiguous ranges in `faces_`
	std::vector<ObjGroup> objects_;

public:
	Model(const char* filename);
	~Model();

	int nverts() const;
	int nfaces() const;

	Eigen::Vector3f vert(int i) const;
	Eigen::Vector2f texCoord(int i) const;
	Eigen::Vector3f normal(int i) const;

	std::vector<VertexIndices> face(int idx) const;
	bool hasNormals() const;

	// NEW: access parsed "o" groups
	const std::vector<ObjGroup>& objects() const { return objects_; }
};

