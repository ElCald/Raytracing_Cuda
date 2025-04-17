// Include
#include "scene.h"

/**
 * @param _camera camera of the scene
 */
__host__ __device__ Scene::Scene(Camera _camera)
    : numTriangles(0), numLights(0), camera(_camera) {}

/**
 * @param triangle Triangle to add
 */
__host__ __device__ void Scene::addTriangle(const Triangle &triangle)
{
    if (numTriangles < MAX_TRIANGLES)
        triangles[numTriangles++] = triangle;
}

/**
 * @param triArray Array of triangles to add
 * @param count Number of triangles
 */
__host__ __device__ void Scene::addTriangles(const Triangle *triArray, int count)
{
    for (int i = 0; i < count && numTriangles < MAX_TRIANGLES; ++i)
        triangles[numTriangles++] = triArray[i];
}

/**
 * @param light Light to add
 */
__host__ __device__ void Scene::addLight(const Light &light)
{
    if (numLights < MAX_LIGHT)
        lights[numLights++] = light;
}

/**
 * @param image image of the scene
 * @param trignales The list of objects in the scene.
 * @param numTriangles The number of objects in the scene.
 * @param camera Camera of the scene.
 * @param lights The list of lights in the scene.
 * @param numLights The number of lights in the scene.
 * function that does the render of the scene
 */
__global__ void renderKernel(Color *__restrict__ image, const Triangle *__restrict__ triangles, int numTriangles, const Camera *__restrict__ camera, const Light *__restrict__ lights, int numLights)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH_PIXEL || y >= HEIGHT_PIXEL)
        return;

    int idx = y * WIDTH_PIXEL + x;

    // Charger localement les données caméra (plus rapide que memory access)
    Ray ray = camera->generateRay(x, y);

    double t, tMin = 1e20;
    int closestIdx = -1;

    // Chercher l'intersection la plus proche
    for (int i = 0; i < numTriangles; ++i)
    {
        if (triangles[i].intersection(ray, t) && t < tMin)
        {
            tMin = t;
            closestIdx = i;
        }
    }

    // Si intersection trouvée, calculer la couleur
    if (closestIdx == -1)
    {
        image[idx] = Color(0, 0, 0);
        return;
    }

    const Triangle &tri = triangles[closestIdx];
    Point3D hitPoint = ray.at(tMin);
    Vecteur3D normal = tri.getNormal(hitPoint);
    Vecteur3D viewDir = (camera->position - hitPoint).normalized();
    Vecteur3D colorVec = phongShading(hitPoint, normal, viewDir, lights, numLights, tri.mat);

    image[idx] = Color(
        static_cast<int>(fmin(255.0, colorVec.x * 255.0)),
        static_cast<int>(fmin(255.0, colorVec.y * 255.0)),
        static_cast<int>(fmin(255.0, colorVec.z * 255.0)));
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
__device__ Vecteur3D phongShading(const Point3D &point, const Vecteur3D &normal, const Vecteur3D &viewDir, const Light *__restrict__ lights, int numLights, const Material &material)
{
    Vecteur3D ambient(0, 0, 0), diffuse(0, 0, 0), specular(0, 0, 0);

    for (int i = 0; i < numLights; ++i)
    {
        Light light = lights[i];

        Vecteur3D rawLightDir = light.position - point;
        double distance2 = rawLightDir.dot(rawLightDir);
        Vecteur3D lightDir = rawLightDir / sqrt(distance2); // Normalize

        double d = sqrt(distance2);
        double attenuation = 1.0 / (1.0 + 0.1 * d + 0.01 * d * d);
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
