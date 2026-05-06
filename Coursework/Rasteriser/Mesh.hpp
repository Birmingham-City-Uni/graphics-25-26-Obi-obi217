#pragma once
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <array>
#include <string>
#include <stdexcept>

struct MeshPart {
	std::string name;
	int startFace = 0;
};

struct Mesh {
	std::vector<Eigen::Vector3f> verts;
	std::vector<Eigen::Vector3f> norms;
	std::vector<Eigen::Vector2f> texs;
	std::vector<std::array<unsigned int, 3>> vFaces;
	std::vector<std::array<unsigned int, 3>> nFaces;
	std::vector<std::array<unsigned int, 3>> tFaces;

	// NEW: for each face in vFaces/nFaces/tFaces, store the active `usemtl` name.
	std::vector<std::string> faceMaterials;

	std::vector<MeshPart> parts;
};

Mesh loadMeshFile(const std::string& filename)
{
	Mesh mesh;

	std::ifstream file(filename);
	if (file.fail()) throw std::runtime_error("Unable to find mesh file: " + filename);

	std::string currentMaterial; // empty means "no material specified"

	std::string line;
	while (!file.eof())
	{
		std::getline(file, line);
		std::stringstream lineSS(line.c_str());
		char lineStart;
		lineSS >> lineStart;
		char ignoreChar;

		if (lineStart == 'o') {
			MeshPart p;
			lineSS >> p.name;
			p.startFace = (int)mesh.vFaces.size();
			mesh.parts.push_back(p);
		}
		else if (lineStart == 'v') {
			if (line.size() > 1 && line[1] == ' ') {
				Eigen::Vector3f v;
				for (int i = 0; i < 3; ++i) lineSS >> v[i];
				mesh.verts.push_back(v);
			}
			else if (line.size() > 1 && line[1] == 'n') {
				lineSS >> ignoreChar;
				Eigen::Vector3f n;
				for (int i = 0; i < 3; ++i) lineSS >> n[i];
				mesh.norms.push_back(n);
			}
			else if (line.size() > 1 && line[1] == 't') {
				lineSS >> ignoreChar;
				Eigen::Vector2f t;
				for (int i = 0; i < 2; ++i) lineSS >> t[i];
				mesh.texs.push_back(t);
			}
		}
		else if (lineStart == 'f') {
			std::array<unsigned int, 3> vFace, nFace, tFace;
			unsigned int idx, idxTex, idxNorm;
			int i = 0;

			while (lineSS >> idx >> ignoreChar >> idxTex >> ignoreChar >> idxNorm) {
				vFace[i] = idx - 1;
				nFace[i] = idxNorm - 1;
				tFace[i] = idxTex - 1;
				++i;
			}

			if (i > 0) {
				mesh.vFaces.push_back(vFace);
				mesh.nFaces.push_back(nFace);
				mesh.tFaces.push_back(tFace);

				// NEW: one entry per pushed face
				mesh.faceMaterials.push_back(currentMaterial);
			}
		}
	}

	// Keep vectors aligned even if a file had no faces.
	if (mesh.faceMaterials.size() != mesh.vFaces.size())
		throw std::runtime_error("Internal error: faceMaterials is out of sync with faces.");

	return mesh;
}