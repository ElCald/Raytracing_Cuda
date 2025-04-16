// Includes
#include "../Geometry/geometry.h"
#include "../GeometricsObjects/forms.h"
#include "../Utils/camera.h"
#include "../Utils/scene.h"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace chrono;

/**
 * @param filename name of the image
 * @param image the image
 * @param width field of view of the image
 * @param height wifth of the image
 */
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

int main(int argc, char *argv[])
{
    // Arguments gestion
    if (argc != 4)
    {
        fprintf(stderr, "Usage : %s [<nbsec> <fps> <nbturns>]\n", argv[0]);
        exit(1);
    }

    int nb_sec = atoi(argv[1]);
    int fps = atoi(argv[2]);
    int nb_turns = atoi(argv[3]);

    if (nb_sec > 300 || fps < 0 || nb_turns < 0)
    {
        fprintf(stderr, "Usage : %s [<nbsec> <fps> <nbtours>]\n", argv[0]);
        fprintf(stderr, "- The number of seconds must not exceed 5min (300 seconds)\n");
        fprintf(stderr, "- The number of fps must be greater than 0 fps\n");
        fprintf(stderr, "- The number of rotations must be greater than 0\n");
        exit(2);
    }

    // Size of the scene
    int width = 1920, height = 1080;

    // Definition of the camera
    Camera cam(Point3D(0, 0, 5), Vecteur3D(0, 0, -1), 90, width, height);

    // Definition of the scene
    Scene scene(cam);

    // Lights definition
    Light *light = new Light(Point3D(-1.5, 0, -5), Vecteur3D(0.5, 1, 0.5)); // White light
    scene.addLight(light);                                                  // Adding the light to lights list

    Light *light2 = new Light(Point3D(1, 0, 5), Vecteur3D(0.4, 0.4, 1)); // White light
    scene.addLight(light2);                                              // Adding the light to lights list

    // Materials definition
    Material materialOrange(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 0.307, 0.168), Vecteur3D(1, 1, 1), 300);

    // Calculating the number of images to be generated
    int nb_images = nb_sec * fps;

    // Cube creation
    Cube *cube = new Cube(3.0, Point3D(0, 0, 0), materialOrange);
    scene.addObject(cube->cube);

    // Definition of cube's center
    Point3D centre_cube = cube->getCenter();
    centre_cube.x += cube->getSize() / 2;
    centre_cube.y += cube->getSize() / 2;
    centre_cube.z -= cube->getSize() / 2;

    // Commands to clean video folder
    int ret = system("rm -r ../build/video/*.ppm");
    ret = system("rm ../build/output.mp4");

    // Creation of a buffer to execute commands after
    char buffer[32];

    // Time measure
    auto t_avant = high_resolution_clock::now(); // Start

    for (int i = 0; i < nb_images; i++)
    {
        // Name of the ouput image
        sprintf(buffer, "video/frame%03d.ppm", i);

        // Turn the cube
        cube->rotateX((180 * nb_turns) / nb_images, cube->getCenter());
        cube->rotateY((360 * nb_turns) / nb_images, cube->getCenter());

        // Creation of the original image (black)
        vector<vector<Color>> image(height, vector<Color>(width, Color(0, 0, 0)));

        // Do the render in the image
        scene.render(image, width, height);

        // Save image
        savePPM(buffer, image, width, height);
    }

    auto t_apres = high_resolution_clock::now(); // End

    // Creation of the video using ffmpeg
    char ffmpegCommand[256];
    sprintf(ffmpegCommand, "ffmpeg -y -framerate %d -i ../build/video/frame%%03d.ppm -c:v libx264 -pix_fmt yuv420p output.mp4", fps);

    // Handle potential error
    ret = system(ffmpegCommand);
    if (ret != 0)
    {
        cerr << "Erreur lors de la création de la vidéo avec ffmpeg." << endl;
    }

    // Computation of the time
    duration<double> t_total = t_apres - t_avant;

    // Final print
    cout << "Images generated: video/framexxx.ppm, with an average fps of : " << (nb_images / t_total.count()) << " (" << t_total.count() << "s)" << endl;

    return 0;
}