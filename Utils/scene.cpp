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

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Ray ray = camera.generateRay(x, y);

            // Variable to store the intersection
            double t, tMin = INFINITY;
            Form *closestObj = nullptr;

            for (auto obj : objects)
            {
                // If intersection with object
                if (obj->intersection(ray, t) && t < tMin)
                {
                    tMin = t;
                    closestObj = obj;
                }
            }

            if (closestObj)
            {
                // The intersection point on the object
                Point3D hitPoint = ray.origine + ray.direction * tMin;

                // Calculate the normal to the object at this point
                Vecteur3D normal = closestObj->getNormal(hitPoint);

                // The direction of view (from the camera towards the point of impact)
                Vecteur3D viewDir = (camera.position - hitPoint).normalized();

                // Colour calculation using the Phong model
                Vecteur3D phongColor = phongShading(hitPoint, normal, viewDir, lights, closestObj->materiau);

                // Apply the calculated colour to the image
                image[y][x] = Color(
                    std::min(255.0, phongColor.x * 255),
                    std::min(255.0, phongColor.y * 255),
                    std::min(255.0, phongColor.z * 255));
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
Vecteur3D phongShading(const Point3D &point, const Vecteur3D &normal, const Vecteur3D &viewDir, std::vector<Light *> lights, const Material &material)
{
    Vecteur3D ambient, diffuse, specular;

    for (auto light : lights)
    {
        // Direction from the point to the light source
        Vecteur3D lightDir = light->position - point;
        lightDir = lightDir.normalized();

        // Reflection vector based on the light direction and surface normal
        Vecteur3D reflectDir = (normal * (2.0 * normal.dot(lightDir))) - lightDir;

        // Distance between the point and the light source
        double distance = std::sqrt(lightDir.x * lightDir.x + lightDir.y * lightDir.y + lightDir.z * lightDir.z);

        // Light attenuation based on distance (quadratic attenuation)
        double attenuation = 1.0 / (distance * distance);

        // Adjust light intensity based on attenuation
        Vecteur3D lightIntensity = light->intensity * attenuation;

        // Ambient component
        ambient = ambient + (material.ambient * lightIntensity);

        // Diffuse component using Lambertian reflection
        double diff = std::max(normal.dot(lightDir), 0.0);
        diffuse = diffuse + (material.diffuse * (lightIntensity * diff));

        // Specular component using Phong model
        double spec = std::pow(std::max(viewDir.dot(reflectDir), 0.0), material.shininess);
        specular = specular + (material.specular * (lightIntensity * spec));
    }

    // Return the total lighting effect
    return ambient + diffuse + specular;
}
