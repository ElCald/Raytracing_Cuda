// Include
#include "scene.h"

/**
 * @param _camera camera of the scene
 */
__host__ __device__ Scene::Scene(Camera &_camera) : camera(_camera), numLights(0), numObjects(0) {}

// Deletor
__host__ __device__ Scene::~Scene()
{
    for (int i = 0; i < numObjects; ++i)
        delete objects[i];
}

/**
 * @param obj vector of triangle we want to add the the scene
 */
__host__ __device__ void Scene::addObject(vector<Triangle *> &obj)
{
    for (auto o : obj)
    {
        if (numObjects < MAX_OBJECTS)
        {
            objects[numObjects++] = o;
        }
    }
}

/**
 * @param obj object we want to add to the scene
 */
__host__ __device__ void Scene::addObject(Form *obj)
{
    if (numObjects < MAX_OBJECTS)
    {
        objects[numObjects++] = obj;
    }
}

/**
 * @param light light we want to add to the scene
 */
__host__ __device__ void Scene::addLight(const Light &light)
{
    if (numLights < MAX_LIGHT)
    {
        lights[numLights++] = light; // Copie directe de l'objet
    }
}

/**
 * @return number of objects in the scene
 */
__host__ __device__ int Scene::getObjectCount() const
{
    return numObjects;
}

/**
 * @return number of lights in the scene
 */
__host__ __device__ int Scene::getLightCount() const
{
    return numLights;
}

/**
 * @return pointer to objects array of the scene
 */
__host__ __device__ Form **Scene::getObjects()
{
    return objects;
}

/**
 * @return pointer to lights array of the scene
 */
__host__ __device__ Light *Scene::getLights()
{
    return lights;
}

/**
 * @param image image of the scene
 * @param objects The list of objects in the scene.
 * @param numObjects The number of objects in the scene.
 * @param camera Camera of the scene.
 * @param lights The list of lights in the scene.
 * @param numLights The number of lights in the scene.
 * function that does the render of the scene
 */
__global__ void renderKernel(Color *image, CubeGPU *cube, int numObjects, CameraGPU *camera, LightGPU *lights, int numLights)
{
    Camera cam = Camera(camera->position, camera->direction, camera->fov, camera->width, camera->height);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH_PIXEL || y >= HEIGHT_PIXEL)
        return;

    int idx = y * WIDTH_PIXEL + x;

    Ray ray = cam.generateRay(x, y);

    /*double t, tMin = INFINITY;
    Form *closestObj = nullptr;

    for (int i = 0; i < numObjects; ++i)
    {
        Form *obj = objects[i];
        if (obj->intersection(ray, t) && t < tMin)
        {
            tMin = t;
            closestObj = obj;
        }
    }*/

    Cube closestObj(cube->size, cube->center, cube->material);

    Point3D hitPoint = ray.origine + ray.direction;
    Vecteur3D normal = closestObj.getNormal(hitPoint);
    Vecteur3D viewDir = (cam.position - hitPoint).normalized();

    Vecteur3D colorVec = phongShading(hitPoint, normal, viewDir, lights, numLights, closestObj.materiau);

    image[idx] = Color(
        static_cast<int>(min(255.0, colorVec.x * 255.0)),
        static_cast<int>(min(255.0, colorVec.y * 255.0)),
        static_cast<int>(min(255.0, colorVec.z * 255.0)));
}

/**
 * @param point     The 3D point on the surface where shading is computed.
 * @param normal    The surface normal at the given point.
 * @param viewDir   The direction vector from the point toward the camera (viewer).
 * @param lights    The list of lights in the scene.
 * @param numLights The number of lights in the scene.
 * @param material  The material properties of the surface (ambient, diffuse, specular, shininess).
 * @return          The resulting color vector from the Phong illumination model.
 * * Computes the Phong shading at a given point using ambient, diffuse, and specular components.
 */
__device__ Vecteur3D phongShading(const Point3D &point, const Vecteur3D &normal, const Vecteur3D &viewDir, LightGPU *lights, int numLights, const Material &material)
{
    Vecteur3D ambient(0, 0, 0), diffuse(0, 0, 0), specular(0, 0, 0);

    for (int i = 0; i < numLights; ++i)
    {
        Light light(lights->position, lights->intensity);

        Vecteur3D rawLightDir = light.position - point;
        double distance2 = rawLightDir.dot(rawLightDir);
        Vecteur3D lightDir = rawLightDir / sqrt(distance2); // Normalize

        double attenuation = 1.0 / distance2;
        Vecteur3D lightIntensity = light.intensity * attenuation;

        // Ambient
        ambient += material.ambient * lightIntensity;

        // Diffuse
        double diff = fmax(normal.dot(lightDir), 0.0);
        diffuse += material.diffuse * (lightIntensity * diff);

        // Specular
        Vecteur3D reflectDir = (normal * (2.0 * normal.dot(lightDir))) - lightDir;
        double spec = pow(fmax(viewDir.dot(reflectDir), 0.0), material.shininess);
        specular += material.specular * (lightIntensity * spec);
    }

    return ambient + diffuse + specular;
}
