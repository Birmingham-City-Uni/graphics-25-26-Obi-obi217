#include <Eigen/Dense>
#include <lodepng.h>
#include <json/json.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "BVHNode.hpp"
#include "Triangle.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "PointLight.hpp"
#include "DirectionalLight.hpp"
#include "SpotLight.hpp"
#include "LambertianShader.hpp"
#include "TexturedLambertianShader.hpp"
#include "PhongShader.hpp"
#include "GlassShader.hpp"
#include "MirrorShader.hpp"
#include "TexCoordTestShader.hpp"
#include "Model.hpp"
#include <fstream>

/// <summary>
/// Load a JSON config file using the nlohmann library.
/// </summary>
nlohmann::json loadConfig(const std::string& filename)
{
	std::ifstream configStream(filename);
	nlohmann::json config = nlohmann::json::parse(configStream);
	return config;
}

struct PngTex {
	std::vector<uint8_t> rgba;
	unsigned w = 0, h = 0;
};

auto loadPng = [](const char* path) {
	PngTex t;
	unsigned err = lodepng::decode(t.rgba, t.w, t.h, path);
	if (err) throw std::runtime_error(std::string("lodepng: ") + lodepng_error_text(err));
	return t;
};

/// <summary>
/// Load an Eigen Vector3f from a config file.
/// Call as for example loadVec3FromConfig(config["myVector3"]);
/// </summary>
Eigen::Vector3f loadVec3FromConfig(const nlohmann::json& config)
{
	return Eigen::Vector3f(config[0], config[1], config[2]);
}

int main(int argc, char* argv[]) {

	// *** Load the config file ***
	auto config = loadConfig("../config/config.json");

	const int pixHeight = config["pixHeight"], pixWidth = config["pixWidth"];
	const int nChannels =4;

	// Optional quality settings (defaults keep old behaviour)
	const int samplesPerPixel = config.value("samplesPerPixel",1);

	// *** Set up camera and output image ***
	Camera cam(
		loadVec3FromConfig(config["cameraPos"]),
		loadVec3FromConfig(config["cameraForward"]),
		loadVec3FromConfig(config["cameraUp"]),
		pixWidth, pixHeight,
		config["cameraFov"]);

	// Optional DoF: if keys are missing, DoF stays disabled.
	if (config.contains("cameraAperture") && config.contains("cameraFocusDist")) {
		cam.setDepthOfField(config["cameraAperture"], config["cameraFocusDist"]);
	}


	std::vector<uint8_t> outImage(pixHeight * pixWidth * nChannels);

	Eigen::Vector3f
		red(1.f, 0.f, 0.f),
		blue(0.f, 0.f, 1.f),
		aqua(0.f, .8f, .8f),
		lavender(178.f / 255.f, 164.f / 255.f, 212.f / 255.f);

	// *** Load shaders and textures ***
	PngTex texTrash = loadPng("../models/blinn1_Base_color.png");
	PngTex texPlane = loadPng("../models/coast_sand_01_diff_2k.png");
	PngTex texEye = loadPng("../models/Eye_BaseColor_tga.png");
	PngTex texMask = loadPng("../models/Char_cyber_Mask_BaseColor.png");
	PngTex texHead = loadPng("../models/Char_cyber_Head_BaseColor.png");
	PngTex texChest = loadPng("../models/Char_cyber_Chest1_BaseColor.png");
	PngTex texLeg = loadPng("../models/Char_cyber_Legs1_BaseColor.png");
	PngTex texOG = loadPng("../models/Buick_GNX_1987_Grille2A_DiffuseAOSO.png");
	PngTex texOne = loadPng("../models/One.png");
	PngTex texTwo = loadPng("../models/Two.png");
	PngTex texThree = loadPng("../models/Buick_GNX_1987_ManufacturerPlateA_Diffuse.png");
	PngTex texFour = loadPng("../models/Buick_GNX_1987_Grille5A_DiffuseAOSO.png");
	PngTex texFive = loadPng("../models/Buick_GNX_1987_InteriorA_Emissive.png");
	PngTex texSix = loadPng("../models/Global_Texture_Coloured_Diffuse_P.png");
	PngTex texSeven = loadPng("../models/Buick_GNX_1987_Grille3A_DiffuseAOSO.png");
	PngTex texEight = loadPng("../models/Buick_GNX_1987_Grille4A_DiffuseAOSO.png");
	PngTex texNine = loadPng("../models/Buick_GNX_1987_TexturedA_Diffuse.png");
	PngTex texTen = loadPng("../models/BadgeA_DiffuseAOSO.png");
	PngTex texEleven = loadPng("../models/One.png");
	PngTex texTwelve = loadPng("../models/One.png");
	PngTex texThirteen = loadPng("../models/Thirteen.png");
	PngTex texFourteen = loadPng("../models/Fourteen.png");
	PngTex texFifteen = loadPng("../models/EngineA_DiffuseAOSO.png");
	PngTex texSixteen = loadPng("../models/Buick_GNX_1987_Textured2A_Diffuse.png");
	PngTex texSeventeen = loadPng("../models/Buick_GNX_1987_InteriorTillingA_DiffuseAOSO.png");
	PngTex texEighteen = loadPng("../models/Buick_GNX_1987_LightA_Emissive.png");
	PngTex texNineteen = loadPng("../models/Buick_GNX_1987_Grille1A_DiffuseAOSO.png");
	PngTex texTwenty = loadPng("../models/InteriorA_DiffuseAOSO.png");
	PngTex texTwentyOne = loadPng("../models/Buick_GNX_1987_LightA_Diffuse.png");
	PngTex texTwentyTwo = loadPng("../models/Twentytwo.png");
	PngTex texTwentyThree = loadPng("../models/Buick_GNX_1987_Grille6A_DiffuseAOSO.png");
	PngTex texWheel = loadPng("../models/Wheel1A_DiffuseAOSO.png");
	PngTex texCalliperZone = loadPng("../models/Buick_GNX_1987_CalliperA_Zone_Diffuse.png");
	PngTex texBackground = loadPng("../models/Background.png");


	LambertianShader redLambertianShader(red);
	//PhongShader bluePlasticShader(blue, Eigen::Vector3f(1.f,1.f,1.f),100.f);
	//TexturedLambertianShader spotShader(&spotTexture, width, height);
	MirrorShader mirrorShader;
	TexCoordTestShader texCoordTestShader;

	//TexturedLambertianShader shTrash(&texTrash.rgba, texTrash.w, texTrash.h);
	PhongShader shTrash(&texTrash.rgba, texTrash.w, texTrash.h, Eigen::Vector3f(1.f, 1.f, 1.f), 200.f);
	TexturedLambertianShader shPlane(&texPlane.rgba, texPlane.w, texPlane.h);
	TexturedLambertianShader shEye(&texEye.rgba, texEye.w, texEye.h);
	TexturedLambertianShader shMask(&texMask.rgba, texMask.w, texMask.h);
	TexturedLambertianShader shHead(&texHead.rgba, texHead.w, texHead.h);
	TexturedLambertianShader shChest(&texChest.rgba, texChest.w, texChest.h);
	TexturedLambertianShader shLeg(&texLeg.rgba, texLeg.w, texLeg.h);
	TexturedLambertianShader shOG(&texOG.rgba, texOG.w, texOG.h);
	TexturedLambertianShader shOne(&texOne.rgba, texOne.w, texOne.h);
	GlassShader shTwo(1.f, Eigen::Vector3f(1.f, 1.f, 1.f), 50);
	TexturedLambertianShader shThree(&texThree.rgba, texThree.w, texThree.h);
	TexturedLambertianShader shFour(&texFour.rgba, texFour.w, texFour.h);
	TexturedLambertianShader shFive(&texFive.rgba, texFive.w, texFive.h);
	TexturedLambertianShader shSix(&texSix.rgba, texSix.w, texSix.h);
	TexturedLambertianShader shSeven(&texSeven.rgba, texSeven.w, texSeven.h);
	TexturedLambertianShader shEight(&texEight.rgba, texEight.w, texEight.h);
	TexturedLambertianShader shNine(&texNine.rgba, texNine.w, texNine.h);
	TexturedLambertianShader shTen(&texTen.rgba, texTen.w, texTen.h);
	TexturedLambertianShader shEleven(&texEleven.rgba, texEleven.w, texEleven.h);
	TexturedLambertianShader shTwelve(&texTwelve.rgba, texTwelve.w, texTwelve.h);
	TexturedLambertianShader shThirteen(&texThirteen.rgba, texThirteen.w, texThirteen.h);
	TexturedLambertianShader shFourteen(&texFourteen.rgba, texFourteen.w, texFourteen.h);
	TexturedLambertianShader shFifteen(&texFifteen.rgba, texFifteen.w, texFifteen.h);
	TexturedLambertianShader shSixteen(&texSixteen.rgba, texSixteen.w, texSixteen.h);
	TexturedLambertianShader shSeventeen(&texSeventeen.rgba, texSeventeen.w, texSeventeen.h);
	TexturedLambertianShader shEighteen(&texEighteen.rgba, texEighteen.w, texEighteen.h);
	TexturedLambertianShader shNineteen(&texNineteen.rgba, texNineteen.w, texNineteen.h);
	TexturedLambertianShader shTwenty(&texTwenty.rgba, texTwenty.w, texTwenty.h);
	TexturedLambertianShader shTwentyOne(&texTwentyOne.rgba, texTwentyOne.w, texTwentyOne.h);
	TexturedLambertianShader shTwentyTwo(&texTwentyTwo.rgba, texTwentyTwo.w, texTwentyTwo.h);
	TexturedLambertianShader shTwentyThree(&texTwentyThree.rgba, texTwentyThree.w, texTwentyThree.h);
	TexturedLambertianShader shWheel(&texWheel.rgba, texWheel.w, texWheel.h);
	TexturedLambertianShader shCalliperZone(&texCalliperZone.rgba, texCalliperZone.w, texCalliperZone.h);
	TexturedLambertianShader shBackground(&texBackground.rgba, texBackground.w, texBackground.h);

	// *** Set up scene ***
	Scene scene;

	// Optional code: here's how to add the spot mesh to the scene, using a BVH
	// Try enabling this and comparing it to the non-BVH version below!
	Model sceneModel("../models/entire scene.obj");
	//scene.renderables.push_back(std::make_shared<BVHNode>(spotModel, &spotShader, 4, rotateY(M_PI / 4.0f)));

	Eigen::Matrix4f modelXform = rotateY(M_PI / 4.0f);

	// For every OBJ `o name` group...
	for (const auto& obj : sceneModel.objects())
	{
		const std::string& name = obj.name;

		// 1) choose shader by *exact* object name (copied from rasteriser mapping)
		const Shader* chosen = &shPlane; // default/fallback (pick anything sensible)

		if (name == "trashpile_3_low.001") chosen = &shTrash;
		else if (name == "Old_trash") chosen = &shTrash;
		else if (name == "Plane") chosen = &shPlane;
		else if (name == "Eye_Inner.Low__Mask.Test.001") chosen = &shEye;
		else if (name == "Eye_Inner.Low__Mask.Test.005") chosen = &shChest;
		else if (name == "Eye_Inner.Low__Mask.Test") chosen = &shMask;
		else if (name == "Eye_Inner.Low__Mask.Test.004") chosen = &shLeg;
		else if (name == "Eye_Inner.Low__Mask.Test.003") chosen = &shHead;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material") chosen = &shOG;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.001") chosen = &shOne;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.002") chosen = &shTwo;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.003") chosen = &shThree;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.004") chosen = &shFour;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.005") chosen = &shFive;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.006") chosen = &shSix;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.007") chosen = &shSeven;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.008") chosen = &shEight;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.009") chosen = &shNine;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.010") chosen = &shTen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.011") chosen = &shEleven;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.012") chosen = &shTwelve;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.013") chosen = &shThirteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.014") chosen = &shFourteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.015") chosen = &shFifteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.016") chosen = &shSixteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.017") chosen = &shSeventeen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.018") chosen = &shEighteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.019") chosen = &shNineteen;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.020") chosen = &shTwenty;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.021") chosen = &shTwentyOne;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.022") chosen = &shTwentyTwo;
		else if (name == "GNX:WindowInside_Geo_lodA_WindowInside_Geo_lodA_Buick_GNX_1987WindowInside_Material.023") chosen = &shTwentyThree;
		else if (name == "polySurface1") chosen = &shWheel;
		else if (name == "polySurface331") chosen = &shWheel;
		else if (name == "polySurface661") chosen = &shWheel;
		else if (name == "polySurface979") chosen = &shWheel;
		else if (name == "polySurface980") chosen = &shCalliperZone;
		else if (name == "polySurface1087") chosen = &shCalliperZone;
		else if (name == "polySurface1194") chosen = &shCalliperZone;
		else if (name == "polySurface1400") chosen = &shCalliperZone;
		else if (name == "Plane.001") chosen = &shBackground;

		// 2) build face list for ONLY this object
		std::vector<std::vector<VertexIndices>> faceList;
		faceList.reserve(obj.faceCount);

		for (int f = obj.faceStart; f < obj.faceStart + obj.faceCount; ++f)
		{
			std::vector<VertexIndices> tri;
			tri.reserve(3);
			tri.push_back(sceneModel.face(f)[0]);
			tri.push_back(sceneModel.face(f)[1]);
			tri.push_back(sceneModel.face(f)[2]);
			faceList.push_back(tri);
		}

		if (faceList.empty())
			continue;

		// 3) add BVH for that object only, using that shader
		scene.renderables.push_back(
			std::make_shared<BVHNode>(sceneModel, chosen, 5, modelXform, &faceList)
		);
	}

	// Here's how to add the mesh without using the BVH.
	// Try comparing performance to the BVH version above.
	//Model spotModel("../models/spot.obj");
	//scene.renderables.push_back(std::make_shared<Mesh>(&spotShader, &spotModel));
	//scene.renderables.back()->modelToWorld(rotateY(M_PI / 4.0f));

	// *** Add lights to scene ***
	Eigen::Vector3f ambientLight(.1f, .1f, .1f);

	std::vector<std::unique_ptr<Light>> lightSources;
	lightSources.push_back(std::make_unique<PointLight>(Eigen::Vector3f(-1.f, 3.f, -1.f), 3.f * Eigen::Vector3f(1.f, 1.f, 1.f)));
	lightSources.push_back(std::make_unique<DirectionalLight>(Eigen::Vector3f(0.f, -1.f, 1.f), .5f * Eigen::Vector3f(1.f, 1.f, 1.f)));
	lightSources.push_back(std::make_unique<SpotLight>(Eigen::Vector3f(1.f, 1.f, 1.f), Eigen::Vector3f(-2.5f, 1.f, -5.5f), Eigen::Vector3f(0.f, -1.f, 0.f), 30.f));

	// *** Render the scene ***

	// Shuffling the scanline order gets better CPU usage between threads
	// when some lines take longer to render than others.
	std::vector<unsigned int> scanlines(pixHeight);
	for (int i = 0; i < pixHeight; ++i) scanlines[i] = i;

	if (config["shuffleScanlines"]) {
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(scanlines.begin(), scanlines.end(), g);
	}

	auto startTime = std::chrono::steady_clock::now();

	#pragma omp parallel for
	for (int y =0; y < pixHeight; ++y) {
		// Per-thread RNG (thread_local so each OpenMP worker gets its own engine)
		thread_local std::mt19937 rng([] {
			std::random_device rd;
			std::seed_seq seq{ rd(), rd(), rd(), rd() };
			return std::mt19937(seq);
		}());

		for (int x =0; x < pixWidth; ++x) {

			Eigen::Vector3f accum(0.f,0.f,0.f);
			int hits =0;

			for (int s =0; s < samplesPerPixel; ++s) {
				Ray ray = (samplesPerPixel >1) ? cam.getRay(x, scanlines[y], rng)
					: cam.getRay(x, scanlines[y]);

				HitInfo hitInfo;
				if (scene.intersect(ray,1e-6f,1e6f, hitInfo, VISIBLE_BITMASK)) {
					Eigen::Vector3f color = hitInfo.shader->getColor(
						hitInfo, &scene,
						lightSources, ambientLight,
						0, config["maxBounces"]);

					color.x() = std::min(color.x(),1.f);
					color.y() = std::min(color.y(),1.f);
					color.z() = std::min(color.z(),1.f);

					accum += color;
					++hits;
				}
			}

			int line = (pixHeight - scanlines[y]) -1;
			if (hits >0) {
				Eigen::Vector3f color = accum / static_cast<float>(samplesPerPixel);
				outImage[(x + line * pixWidth) * nChannels +0] = static_cast<uint8_t>(std::min(color.x(),1.f) *255.f);
				outImage[(x + line * pixWidth) * nChannels +1] = static_cast<uint8_t>(std::min(color.y(),1.f) *255.f);
				outImage[(x + line * pixWidth) * nChannels +2] = static_cast<uint8_t>(std::min(color.z(),1.f) *255.f);
				outImage[(x + line * pixWidth) * nChannels +3] =255;
			}
			else {
				outImage[(x + line * pixWidth) * nChannels +0] =0;
				outImage[(x + line * pixWidth) * nChannels +1] =0;
				outImage[(x + line * pixWidth) * nChannels +2] =0;
				outImage[(x + line * pixWidth) * nChannels +3] =255;
			}
		}
		if (omp_get_thread_num() == omp_get_num_threads()-1) {
			std::clog << "\rScanlines remaining: " << (pixHeight - y) << ' ' << std::flush;
		}

	}

	auto renderTime = std::chrono::steady_clock::now() - startTime;

	std::cout << "Render duration " << std::chrono::duration_cast<std::chrono::milliseconds>(renderTime).count() * 1e-3f << " seconds." << std::endl;

	// *** Save the output image ***
	int errorCode;
	errorCode = lodepng::encode(config["outputFilename"], outImage, pixWidth, pixHeight);
	if (errorCode) { // check the error code, in case an error occurred.
		std::cout << "lodepng error encoding image: " << lodepng_error_text(errorCode) << std::endl;
		return errorCode;
	}

	return 0;
}
