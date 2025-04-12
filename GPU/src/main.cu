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
 */
void savePPM(const string &filename, const Color *image)
{
    ofstream file(filename);
    if (!file.is_open())
        return;

    file << "P3\n"
         << WIDTH_PIXEL << " " << HEIGHT_PIXEL << "\n255\n";

    for (int y = 0; y < HEIGHT_PIXEL; y++)
    {
        for (int x = 0; x < WIDTH_PIXEL; x++)
        {
            const Color &color = image[y * WIDTH_PIXEL + x];
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

    // ---- HOST variables ----

    // Calculating the number of images to be generated
    int nb_images = nb_sec * fps;

    // Commands to clean video folder
    int ret = system("rm -r ../build/video/*.ppm");
    ret = system("rm ../build/output.mp4");

    // Creation of a buffer to execute commands after
    char buffer[32];

    // Definition of the camera
    Camera cam(Point3D(0, 0, 5), Vecteur3D(0, 0, -1), 90, WIDTH_PIXEL, HEIGHT_PIXEL);

    // Definition of the scene
    Scene scene(cam);

    // Lights definition
    Light light1(Point3D(-1.5, 0, -5), Vecteur3D(0.5, 1, 0.5));
    scene.addLight(light1);

    Light light2(Point3D(1, 0, 5), Vecteur3D(0.4, 0.4, 1));
    scene.addLight(light2);

    // Materials definition
    Material materialOrange(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 0.307, 0.168), Vecteur3D(1, 1, 1), 300);

    // Cube creation
    Cube *cube = new Cube(3.0, Point3D(0, 0, 0), materialOrange);
    scene.addObject(cube->cube);

    // Definition of cube's center
    Point3D centre_cube = cube->getCenter();
    centre_cube.x += cube->getSize() / 2;
    centre_cube.y += cube->getSize() / 2;
    centre_cube.z -= cube->getSize() / 2;

    // ---- DEVICE variables ----

    // Image
    Color *d_image;
    cudaMalloc(&d_image, WIDTH_PIXEL * HEIGHT_PIXEL * sizeof(Color));

    LightGPU *d_lights;
    cudaMalloc(&d_lights, scene.getLightCount() * sizeof(Light));
    cudaMemcpy(d_lights, scene.getLights(), scene.getLightCount() * sizeof(Light), cudaMemcpyHostToDevice);

    CameraGPU d_cam;
    d_cam.position = cam.position;   // Point3D
    d_cam.direction = cam.direction; // Vecteur3D
    d_cam.fov = cam.fov;
    d_cam.width = cam.width;
    d_cam.height = cam.height;

    CameraGPU *d_camera;
    cudaMalloc(&d_camera, sizeof(CameraGPU));
    cudaMemcpy(d_camera, &d_cam, sizeof(CameraGPU), cudaMemcpyHostToDevice);

    CubeGPU *d_cube;
    cudaMalloc(&d_cube, sizeof(CubeGPU));

    CubeGPU gpuCube(cube->getSize(), cube->getCenter(), cube->materiau);

    // Copie du CubeGPU vers le GPU
    cudaMemcpy(d_cube, &gpuCube, sizeof(CubeGPU), cudaMemcpyHostToDevice);

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
        Color *h_image = new Color[WIDTH_PIXEL * HEIGHT_PIXEL];

        // Initialisation en noir (0,0,0)
        for (int i = 0; i < WIDTH_PIXEL * HEIGHT_PIXEL; ++i)
            h_image[i] = Color(0, 0, 0);

        // Do the render in the image
        renderKernel<<<gridDim, blockDim>>>(d_image, d_cube, scene.getObjectCount(), d_camera, d_lights, scene.getLightCount());

        // Vérification d'erreur
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        cudaMemcpy(h_image, d_image, WIDTH_PIXEL * HEIGHT_PIXEL * sizeof(Color), cudaMemcpyDeviceToHost);

        // Save image
        savePPM(buffer, h_image);

        delete[] h_image;
        cudaFree(d_image);
        cudaFree(d_camera);
        cudaFree(d_lights);
        cudaFree(d_cube);
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