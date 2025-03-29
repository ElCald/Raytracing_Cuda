#include <iostream>
#include <fstream>
#include "../Geometry/geometry.h"
#include "../GeometricObjects/formes.h"
#include "../Cameras/camera.h"

using namespace std;

void savePPM(const string &filename, const vector<vector<Color>> &image, int width, int height)
{
    ofstream file(filename);
    if (!file.is_open())
        return;

    file << "P3\n"
         << width << " " << height << "\n255\n";

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Color color = image[y][x];
            file << color.r << " " << color.g << " " << color.b << " ";
        }
        file << "\n";
    }
    file.close();
}

// Fonction pour calculer la couleur avec Phong
Vecteur3D phongShading(const Point3D &point, const Vecteur3D &normal, const Vecteur3D &viewDir, const Light &light, const Material &material)
{
    Vecteur3D lightDir = (light.position - point).normalized();
    Vecteur3D reflectDir = (normal * (2.0 * normal.dot(lightDir))) - lightDir;

    // Composante ambiante
    Vecteur3D ambient = material.ambient * light.intensity;

    // Composante diffuse (Lambert)
    double diff = std::max(normal.dot(lightDir), 0.0);
    Vecteur3D diffuse = material.diffuse * (light.intensity * diff);

    // Composante spéculaire (Phong)
    double spec = std::pow(std::max(viewDir.dot(reflectDir), 0.0), material.shininess);
    Vecteur3D specular = material.specular * (light.intensity * spec);

    return ambient + diffuse + specular;
}

int main()
{
    int width = 800, height = 600;
    Camera cam(Point3D(0, 0, 0), Vecteur3D(0, 0, -1), 90, width, height);

    Light light(Point3D(5, 5, 0), Vecteur3D(0.7, 0.7, 0.7)); // Lumière blanche en (5,5,0)

    Material materialRed(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 0, 0), Vecteur3D(1, 1, 1), 32);
    Material materialBlue(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0, 0, 1), Vecteur3D(1, 1, 1), 32);
    Material materialGreen(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0, 1, 0), Vecteur3D(1, 1, 1), 32);
    Material materialOrange(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0.94, 0.6, 0.16), Vecteur3D(1, 1, 1), 32);
    Material materialGray(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0.7, 0.7, 0.7), Vecteur3D(1.0, 1.0, 1.0), 32);

    vector<Forme *> objects;

    // Crée une sphère rouge
    // Sphere *sphere = new Sphere(Point3D(0, 0, -5), 1, materialRed);
    // objects.push_back(sphere);

    // Crée une sphère orange
    Sphere *sphere2 = new Sphere(Point3D(0, 0, -4), 1.5, materialBlue);
    objects.push_back(sphere2);

    // Crée un carre bleue
    // Carre *carre = new Carre(Point3D(-4, 0, -5), Point3D(-2, 0, -5), Point3D(-2, 2, -5), Point3D(-4, 2, -5), materialBlue);
    // objects.push_back(carre);

    // Crée un triangle vert
    Triangle *triangle = new Triangle(Point3D(2, 0, -5), Point3D(4, 0, -5), Point3D(3, 2, -5), materialGreen);
    objects.push_back(triangle);

    // Crée un cube gris
    // Cube* cube = new Cube(2.0, Point3D(-4, 0, -5), materialBlue);
    // cube->rotateX(30);
    // cube->rotateY(30);
    // objects.push_back(cube);

    vector<vector<Color>> image(height, vector<Color>(width, Color(0, 0, 0))); // Image noire initiale

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Ray ray = cam.generateRay(x, y);

            // Variable pour stocker l'intersection
            double t, tMin = INFINITY;
            Forme *closestObj = nullptr;

            // image[y][x] = Color(0, 0, 0); // Par défaut, noir

            for (auto obj : objects)
            {
                if (obj->intersection(ray, t) && t < tMin) // Si intersection avec l'objet
                {
                    tMin = t;
                    closestObj = obj;

                    // image[y][x] = obj->couleur; // On assigne la couleur de l'objet
                    // break;                      // Sortir dès qu'on trouve une intersection
                }
            }

            if (closestObj)
            {
                // Le point d'intersection sur l'objet
                Point3D hitPoint = ray.origine + ray.direction * tMin;
                // Calcul de la normale à l'objet à ce point
                Vecteur3D normal = closestObj->getNormal(hitPoint);
                // La direction de la vue (depuis la caméra vers le point d'impact)
                Vecteur3D viewDir = (cam.position - hitPoint).normalized();

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

    savePPM("output.ppm", image, width, height);
    cout << "Image generated: output.ppm\n";

    // Libération de la mémoire
    for (auto obj : objects)
    {
        delete obj;
    }

    return 0;
}
