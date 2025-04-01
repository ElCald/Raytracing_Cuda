// Includes
#include "../Geometry/geometry.h"
#include "../GeometricObjects/formes.h"
#include "../Utils/camera.h"
#include "../Utils/scene.h"
#include <iostream>
#include <fstream>

using namespace std;

// Function to save a .ppm image
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
    // Size of the scene
    int width = 1920, height = 1080;

    // Definition of the camera
    Camera cam(Point3D(1, 0, 5), Vecteur3D(0, 0, -1), 90, width, height);

    // Definition of the scene
    Scene scene(cam);

    // Lights definition
    Light *light = new Light(Point3D(-1.5, 0, -5), Vecteur3D(0, 1, 0)); // White light
    scene.addLight(light);                                              // Adding the light to lights list

    // Materials definition
    Material materialRed(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 0, 0), Vecteur3D(1, 1, 1), 32);
    Material materialBlue(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0, 0, 1), Vecteur3D(1, 1, 1), 32);
    Material materialGreen(Vecteur3D(0.01, 0.01, 0.01), Vecteur3D(0, 1, 0), Vecteur3D(1, 1, 1), 32);
    Material materialYellow(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 1, 0), Vecteur3D(1, 1, 1), 32);
    Material materialPurple(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 0, 1), Vecteur3D(1, 1, 1), 32);
    Material materialOrange(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 0.307, 0.168), Vecteur3D(1, 1, 1), 300);
    Material materialGray(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0.7, 0.7, 0.7), Vecteur3D(1.0, 1.0, 1.0), 32);
    Material materialWhite(Vecteur3D(0.2, 0.2, 0.2), Vecteur3D(1, 1, 1), Vecteur3D(1, 1, 1), 32);

    // Sphere definition
    Sphere *sphere = new Sphere(Point3D(0, 0, -8), 3, materialWhite);
    scene.addObject(sphere); // Adding the sphere to the list of objects

    // Cubes definition
    /*Cube *cube = new Cube(3.0, Point3D(1.5, -3, 0), materialPurple);
    scene.addObject(cube->cube); // Adding the cube to the list of objects
    // Some rotation
    cube->rotateX(30);
    cube->rotateY(30);

    // Same for others cube
    Cube *cube2 = new Cube(2.0, Point3D(-4, 1, -2), materialRed);
    scene.addObject(cube2->cube);

    Cube *cube3 = new Cube(3.0, Point3D(-5, 0.8, -3.5), materialYellow);
    cube3->rotateY(15);
    cube3->rotateZ(30);
    scene.addObject(cube3->cube);

    Cube *cube4 = new Cube(3.0, Point3D(4, -3, -0.5), materialOrange);
    scene.addObject(cube4->cube);
    cube4->rotateX(30);
    cube4->rotateY(20);*/

    // Creation of the original image (black)
    vector<vector<Color>> image(height, vector<Color>(width, Color(0, 0, 0)));

    // Do the render in the image
    scene.render(image, width, height);

    // Save image
    savePPM("output.ppm", image, width, height);
    cout << "Image generated: output.ppm" << endl;

    return 0;
}
