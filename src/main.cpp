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

int main()
{
    int width = 800, height = 600;
    Camera cam(Point3D(0, 0, 0), Vecteur3D(0, 0, -1), 90, width, height);

    vector<Forme *> objects;

    // Crée une sphère rouge
    Sphere *sphere = new Sphere(Point3D(0, 0, -5), 1, Color(255, 0, 0));
    objects.push_back(sphere);

    // Crée un carre bleue
    Carre *carre = new Carre(Point3D(-4, 0, -5), Point3D(-2, 0, -5), Point3D(-2, 2, -5), Point3D(-4, 2, -5), Color(0, 0, 255));
    objects.push_back(carre);

    // Crée un triangle vert
    Triangle *triangle = new Triangle(Point3D(2, 0, -5), Point3D(4, 0, -5), Point3D(3, 2, -5), Color(0, 255, 0));
    objects.push_back(triangle);

    vector<vector<Color>> image(height, vector<Color>(width, Color(0, 0, 0))); // Image noire initiale

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Ray ray = cam.generateRay(x, y);

            // Variable pour stocker l'intersection
            double t;
            image[y][x] = Color(0, 0, 0); // Par défaut, noir

            for (auto obj : objects)
            {
                if (obj->intersection(ray, t)) // Si intersection avec l'objet
                {
                    image[y][x] = obj->couleur; // On assigne la couleur de l'objet
                    break;                      // Sortir dès qu'on trouve une intersection
                }
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
