// Includes
#include "../Geometry/geometry.h"
#include "../GeometricObjects/formes.h"
#include "../Utils/camera.h"
#include "../Utils/scene.h"
#include <iostream>
#include <fstream>
#include <chrono>

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
    Camera cam(Point3D(0, 0, 5), Vecteur3D(0, 0, -1), 90, width, height);

    // Definition of the scene
    Scene scene(cam);

    // Lights definition
    Light *light = new Light(Point3D(-1.5, 0, -5), Vecteur3D(0.5, 1, 0.5)); // White light
    scene.addLight(light);                                              // Adding the light to lights list

    Light *light2 = new Light(Point3D(1, 0, 5), Vecteur3D(0.4, 0.4, 1)); // White light
    scene.addLight(light2);                                              // Adding the light to lights list

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
    // Sphere *sphere = new Sphere(Point3D(0, 0, -8), 3, materialWhite);
    // scene.addObject(sphere); // Adding the sphere to the list of objects

    // Cubes definition
    /*Cube *cube = new Cube(3.0, Point3D(1.5, -3, 0), materialPurple);
    scene.addObject(cube->cube); // Adding the cube to the list of objects
    // Some rotation
    cube->rotateX(30, cube->getCenter());
    cube->rotateY(30, cube->getCenter());

    // Same for others cube
    Cube *cube2 = new Cube(2.0, Point3D(-4, 1, -2), materialRed);
    scene.addObject(cube2->cube);

    Cube *cube3 = new Cube(3.0, Point3D(-5, 0.8, -3.5), materialYellow);
    cube3->rotateY(15, cube3->getCenter());
    cube3->rotateZ(30, cube3->getCenter());
    scene.addObject(cube3->cube);

    Cube *cube4 = new Cube(3.0, Point3D(4, -3, -0.5), materialOrange);
    scene.addObject(cube4->cube);
    cube4->rotateX(30, cube4->getCenter());
    cube4->rotateY(20, cube4->getCenter());*/

/*
    Cube *cube5 = new Cube(3.0, Point3D(0, -2, -3.5), materialOrange);
    scene.addObject(cube5->cube);

    
    // Creation of the original image (black)
    vector<vector<Color>> image(height, vector<Color>(width, Color(0, 0, 0)));

    // Do the render in the image
    scene.render(image, width, height);


    // Save image
    savePPM("output.ppm", image, width, height);
    cout << "Image generated: output.ppm" << endl;
*/

    Cube *cube5 = new Cube(3.0, Point3D(0, 0, 0), materialOrange);
    scene.addObject(cube5->cube);
    


    char buffer[32];

    int nb_tour = 2;
    int nb_images = 120;

    Point3D centre_cube = cube5->getCenter();

    centre_cube.x += cube5->getSize()/2;
    centre_cube.y += cube5->getSize()/2;
    centre_cube.z -= cube5->getSize()/2;


    auto t_avant = chrono::high_resolution_clock::now();  // Début


    for(int i=0; i<nb_images; i++){

        // nom image sortie
        sprintf(buffer, "video/frame%03d.ppm", i);


        cube5->rotateX((360*nb_tour)/nb_images, cube5->getCenter());
        cube5->rotateY((360*nb_tour)/nb_images, cube5->getCenter());

        // Creation of the original image (black)
        vector<vector<Color>> image(height, vector<Color>(width, Color(0, 0, 0)));

        // Do the render in the image
        scene.render(image, width, height);

        // Save image
        savePPM(buffer, image, width, height);

    }

    auto t_apres = chrono::high_resolution_clock::now();  // Fin

    chrono::duration<double> t_total = t_apres - t_avant;

    cout << "Images generated: video/framexxx.ppm, time: " << t_total.count() << endl;

    return 0;
}
