#ifndef SCENE_H
#define SCENE_H

#include "../GeometricsObjects/forms.h"
#include "light.h"
#include "camera.h"

#define HEIGHT_PIXEL 1080
#define WIDTH_PIXEL 1920
#define MAX_TRIANGLES 2048
#define MAX_LIGHT 3

// Scene struct
struct Scene
{
    // Attributes
    Triangle triangles[MAX_TRIANGLES];
    int numTriangles;

    Light lights[MAX_LIGHT];
    int numLights;

    Camera camera;

    // Methods
    __host__ __device__ Scene(Camera _camera);
    __host__ __device__ void addTriangle(const Triangle &triangle);
    __host__ __device__ void addTriangles(const Triangle *triArray, int count);
    __host__ __device__ void addLight(const Light &light);
};

// Kernel
__global__ void renderKernel(Color *__restrict__ image, const Triangle *__restrict__ triangles, int numTriangles, const Camera *__restrict__ camera, const Light *__restrict__ lights, int numLights);

// Phong function
__device__ Vecteur3D phongShading(const Point3D &point, const Vecteur3D &normal, const Vecteur3D &viewDir, const Light *__restrict__ lights, int numLights, const Material &material);

#endif // SCENE_H
