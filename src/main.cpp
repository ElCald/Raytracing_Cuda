#include <iostream>
#include <fstream>
#include "../Geometry/geometry.h"
#include "../GeometricObjects/formes.h"
#include "../Cameras/camera.h"
#include "../Utils/scene.h"

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

    Camera cam(Point3D(1, 0, 5), Vecteur3D(0, 0, -1), 90, width, height);

    Light light(Point3D(0, 1.5, 5), Vecteur3D(0.8, 0.8, 0.8)); // Lumière blanche 

    Scene scene(cam, light);

    Material materialRed(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 0, 0), Vecteur3D(1, 1, 1), 32);
    Material materialBlue(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0, 0, 1), Vecteur3D(1, 1, 1), 32);
    Material materialGreen(Vecteur3D(0.01, 0.01, 0.01), Vecteur3D(0, 1, 0), Vecteur3D(1, 1, 1), 32);
    Material materialYellow(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 1, 0), Vecteur3D(1, 1, 1), 32);
    Material materialPurple(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 0, 1), Vecteur3D(1, 1, 1), 32);
    Material materialOrange(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 0.307, 0.168), Vecteur3D(1, 1, 1), 300);
    Material materialGray(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0.7, 0.7, 0.7), Vecteur3D(1.0, 1.0, 1.0), 32);



    // Crée un carre bleue
    // Carre *carre = new Carre(Point3D(-4, 0, -5), Point3D(-2, 0, -5), Point3D(-2, 2, -5), Point3D(-4, 2, -5), materialBlue);
    // scene.addObject(carre);

    // Crée un triangle vert
    // Triangle *triangle = new Triangle(Point3D(1, 1, -5), Point3D(5, 1, -5), Point3D(3, 4, -5), materialYellow);
    // scene.addObject(triangle);


    Sphere *sphere = new Sphere(Point3D(0, 0, -8), 1, materialBlue);
    scene.addObject(sphere);

    Cube *cube = new Cube(3.0, Point3D(1.5, -3, 0), materialPurple);
    scene.addObject(cube->cube);
    cube->rotateX(30);
    cube->rotateY(30);

    Cube *cube2 = new Cube(2.0, Point3D(-4, 1, -2), materialRed);
    scene.addObject(cube2->cube);

    Cube *cube3 = new Cube(3.0, Point3D(-5, 0.8, -3.5), materialYellow);
    cube3->rotateY(15);
    cube3->rotateZ(30);
    scene.addObject(cube3->cube);

    Cube *cube4 = new Cube(3.0, Point3D(4, -3, -0.5), materialOrange);
    scene.addObject(cube4->cube);
    cube4->rotateX(30);
    cube4->rotateY(20);


    vector<vector<Color>> image(height, vector<Color>(width, Color(0, 0, 0))); // Image noire initiale


    scene.render(image, width, height); // Rendu de la scène dans l'image


    savePPM("output.ppm", image, width, height);
    cout << "Image generated: output.ppm\n";


    return 0;
}
