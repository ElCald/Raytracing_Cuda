// scene.h
#ifndef SCENE_H
#define SCENE_H

// Includes
#include "../GeometricsObjects/forms.h"
#include "light.h"
#include "camera.h"
#include <vector>

#define HEIGHT_PIXEL 1080
#define WIDTH_PIXEL 1920
#define MAX_OBJECTS 10
#define MAX_LIGHT 3

using namespace std;

// Scene class
class Scene
{
public:
    // Attributes
    Camera &camera;

    // Constructor and Deletor
    __host__ __device__ Scene(Camera &_camera);
    __host__ __device__ ~Scene();

    // Methods
    __host__ __device__ void addObject(vector<Triangle *> &obj);
    __host__ __device__ void addObject(Form *obj);
    __host__ __device__ void addLight(const Light &light);

    // Getters
    __host__ __device__ int getObjectCount() const;
    __host__ __device__ int getLightCount() const;
    __host__ __device__ Form **getObjects();
    __host__ __device__ Light *getLights();

private:
    // Attribute
    Form *objects[MAX_OBJECTS];
    int numObjects = 0;
    Light lights[MAX_LIGHT];
    int numLights = 0;
};

struct LightGPU
{
    Point3D position;
    Vecteur3D intensity;
};

struct CameraGPU
{
    Point3D position;
    Vecteur3D direction;
    float fov;
    int width, height;
};

struct CubeGPU
{
    double size;
    Point3D center;
    MaterialGPU material;
};

// Kernel
__global__ void renderKernel(Color *image, CubeGPU *cube, int numObjects, CameraGPU *camera, LightGPU *lights, int numLights);

// Phong function
__device__ Vecteur3D phongShading(const Point3D &point, const Vecteur3D &normal, const Vecteur3D &viewDir, Light *lights, int numLights, const Material &material);

#endif // SCENE_H