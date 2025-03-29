#include "scene.h"

Scene::Scene(Camera& _camera, Light& _light) : camera(_camera), light(_light){}


Scene::~Scene() {
    for(auto obj : objects){
        delete obj;
    }
}


void Scene::addObject(vector<Triangle*>& obj)
{
    for(auto o : obj){
        objects.push_back(o);
    }
}


void Scene::addObject(Forme* obj)
{
    objects.push_back(obj);
}


// bool Scene::intersect(const Ray &ray, Point3D &hitPoint, Vecteur3D &normal, int &materiau) const
// {
//     bool hit = false;
//     double minDist = 1e9;
//     for (const auto &obj : objects)
//     {
//         if (obj->intersection(ray.origine))
//         { // Simplification, améliorer la détection
//             hit = true;
//             materiau = obj->materiau;
//         }
//     }
//     return hit;
// }


void Scene::render(vector<vector<Color>>& image, int width, int height){

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Ray ray = camera.generateRay(x, y);

            // Variable pour stocker l'intersection
            double t, tMin = INFINITY;
            Forme *closestObj = nullptr;

            for (auto obj : objects)
            {
                if (obj->intersection(ray, t) && t < tMin) // Si intersection avec l'objet
                {
                    tMin = t;
                    closestObj = obj;
                }
            }

            if (closestObj)
            {
                // Le point d'intersection sur l'objet
                Point3D hitPoint = ray.origine + ray.direction * tMin;
                // Calcul de la normale à l'objet à ce point
                Vecteur3D normal = closestObj->getNormal(hitPoint);
                // La direction de la vue (depuis la caméra vers le point d'impact)
                Vecteur3D viewDir = (camera.position - hitPoint).normalized();

                // Calcul de la couleur avec le modèle de Phong
                Vecteur3D phongColor = phongShading(hitPoint, normal, viewDir, light, closestObj->materiau);

                // Applique la couleur calculée sur l'image
                image[y][x] = Color(
                    std::min(255.0, phongColor.x * 255),
                    std::min(255.0, phongColor.y * 255),
                    std::min(255.0, phongColor.z * 255));
            }
            else
            {
                // Si aucune intersection, fond noir
                image[y][x] = Color(0, 0, 0);
            }
        }
    }

}



// Fonction pour calculer la couleur avec Phong
Vecteur3D phongShading(const Point3D &point, const Vecteur3D &normal, const Vecteur3D &viewDir, const Light &light, const Material &material)
{
    // Direction de la lumière (vecteur entre le point et la source lumineuse)
    Vecteur3D lightDir = light.position - point;  // Soustraction entre Point3D et Point3D donne un Vecteur3D
    lightDir = lightDir.normalized(); // Normaliser pour obtenir un vecteur unitaire

    // Calcul du vecteur de réflexion
    Vecteur3D reflectDir = (normal * (2.0 * normal.dot(lightDir))) - lightDir;

    // Calcul de la distance entre le point d'impact et la source de lumière
    double distance = std::sqrt(lightDir.x * lightDir.x + lightDir.y * lightDir.y + lightDir.z * lightDir.z); // Utilisation de norm() pour calculer la distance

    // Atténuation de la lumière en fonction de la distance (atténuation quadratique)
    double attenuation = 1.0 / (distance * distance); // Atténuation quadratique

    // Appliquer l'atténuation à l'intensité de la lumière
    Vecteur3D lightIntensity = light.intensity * attenuation;

    // Composante ambiante
    Vecteur3D ambient = material.ambient * lightIntensity;

    // Composante diffuse (Lambert)
    double diff = std::max(normal.dot(lightDir), 0.0);
    Vecteur3D diffuse = material.diffuse * (lightIntensity * diff);

    // Composante spéculaire (Phong)
    double spec = std::pow(std::max(viewDir.dot(reflectDir), 0.0), material.shininess);
    Vecteur3D specular = material.specular * (lightIntensity * spec);

    return ambient + diffuse + specular;
}