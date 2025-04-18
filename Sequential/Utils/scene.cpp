// Include
#include "scene.h"

/**
 * @param _camera camera of the scene
 */
Scene::Scene(Camera &_camera) : camera(_camera) {}

// Deletor
Scene::~Scene()
{
    for (auto obj : objects)
    {
        delete obj;
    }

    for (auto light : lights)
    {
        delete light;
    }
}

/**
 * @param obj vector of triangle we want to add the the scene
 */
void Scene::addObject(vector<Triangle *> &obj)
{
    for (auto o : obj)
    {
        objects.push_back(o);
    }
}

/**
 * @param obj object we want to add to the scene
 */
void Scene::addObject(Form *obj)
{
    objects.push_back(obj);
}

/**
 * @param light light we want to add to the scene
 */
void Scene::addLight(Light *light)
{
    lights.push_back(light);
}

/**
 * @param image image of the scene
 * @param width width of the image
 * @param height height of the image
 * function that does the render of the scene
 */
void Scene::render(vector<vector<Color>> &image, int width, int height)
{
    const Point3D camPos = camera.position; // Cached once

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            Ray ray = camera.generateRay(x, y);

            double t, tMin = INFINITY;
            Form *closestObj = nullptr;

            for (auto obj : objects)
            {
                if (obj->intersection(ray, t) && t < tMin)
                {
                    tMin = t;
                    closestObj = obj;
                }
            }

            if (closestObj)
            {
                Point3D hitPoint = ray.origine + ray.direction * tMin;
                Vecteur3D normal = closestObj->getNormal(hitPoint);
                Vecteur3D viewDir = (camPos - hitPoint).normalized();

                Vecteur3D phongColor = phongShading(hitPoint, normal, viewDir, lights, closestObj->materiau);

                // Clamp and convert to int
                int r = static_cast<int>(std::min(255.0, phongColor.x * 255));
                int g = static_cast<int>(std::min(255.0, phongColor.y * 255));
                int b = static_cast<int>(std::min(255.0, phongColor.z * 255));

                image[y][x] = Color(r, g, b);
            }
        }
    }
}

/**
 * @param point     The 3D point on the surface where shading is computed.
 * @param normal    The surface normal at the given point.
 * @param viewDir   The direction vector from the point toward the camera (viewer).
 * @param lights    The list of lights in the scene.
 * @param material  The material properties of the surface (ambient, diffuse, specular, shininess).
 * @return          The resulting color vector from the Phong illumination model.
 * * Computes the Phong shading at a given point using ambient, diffuse, and specular components.
 */
Vecteur3D phongShading(const Point3D &point, const Vecteur3D &normal, const Vecteur3D &viewDir, const std::vector<Light *> &lights, const Material &material)
{
    Vecteur3D ambient(0, 0, 0), diffuse(0, 0, 0), specular(0, 0, 0);

    for (const auto &light : lights)
    {
        // Direction from point to light (non-normalized)
        Vecteur3D rawLightDir = light->position - point;
        double distance2 = rawLightDir.dot(rawLightDir);
        Vecteur3D lightDir = rawLightDir / std::sqrt(distance2); // Normalize

        double d = sqrt(distance2);
        double attenuation = 1.0 / (1.0 + 0.1 * d + 0.01 * d * d);
        Vecteur3D lightIntensity = light->intensity * attenuation;

        // Ambient
        ambient += material.ambient * lightIntensity;

        // Diffuse
        double diff = std::max(normal.dot(lightDir), 0.0);
        diffuse += material.diffuse * (lightIntensity * diff);

        // Specular
        Vecteur3D reflectDir = (normal * (2.0 * normal.dot(lightDir))) - lightDir;
        double spec = std::pow(std::max(viewDir.dot(reflectDir), 0.0), material.shininess);
        specular += material.specular * (lightIntensity * spec);
    }

    return ambient + diffuse + specular;
}
