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
    // -- Gestion des arguments (inchangée) --
    if (argc != 4)
    {
        fprintf(stderr, "Usage : %s [<nbsec> <fps> <nbturns>]\n", argv[0]);
        exit(1);
    }

    int nb_sec = atoi(argv[1]);
    int fps = atoi(argv[2]);
    int nb_turns = atoi(argv[3]);

    if (nb_sec > 300 || fps < 1 || nb_turns < 1)
    {
        fprintf(stderr, "Arguments invalides.\n");
        exit(2);
    }

    int nb_images = nb_sec * fps;
    int ret = system("rm -r ../build/video/*.ppm");
    if (ret != 0)
        cerr << "Erreur lors de la création de la vidéo avec ffmpeg." << endl;
    ret = system("rm ../build/output.mp4");
    if (ret != 0)
        cerr << "Erreur lors de la création de la vidéo avec ffmpeg." << endl;
    char buffer[256];

    // -- Caméra et scène --
    Camera cam(Point3D(0, 0, 5), Vecteur3D(0, 0, -1), 90, WIDTH_PIXEL, HEIGHT_PIXEL, Vecteur3D(1, 0, 0), Vecteur3D(0, 1, 0));
    Scene scene(cam);

    // -- Lumières --
    scene.addLight(Light(Point3D(-1.5, 0, -5), Vecteur3D(0.5, 1, 0.5)));
    scene.addLight(Light(Point3D(1, 0, 5), Vecteur3D(0.4, 0.4, 1)));

    // -- Matériaux --
    Material matOrange(Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(1, 0.307, 0.168), Vecteur3D(1, 1, 1), 300);

    // -- Objet : cube --
    Cube *cube = new Cube(3.0, Point3D(0, 0, 0), matOrange);
    scene.addTriangles(cube->triangles, 12);

    // -- Objet : pyramid --
    /*Pyramid *pyramid = new Pyramid(
        Point3D(-1, -1, 0), // base point 1
        Point3D(1, -1, 0),  // base point 2
        Point3D(0, 1, 0),   // base point 3
        Point3D(0, 0, 2),   // apex
        matOrange, Point3D(0, 0, 0));
    scene.addTriangles(pyramid->triangles, 4);*/

    // -- Objet : sphère --
    /*TriangleSphere *sphere = new TriangleSphere(Point3D(0, 0, 0));
    sphere->generate(Point3D(0, 0, 0), 1.5, 20, 20, matOrange);

    scene.addTriangles(sphere->triangles, sphere->count);*/

    // -- Triangle device --
    Triangle *d_triangles;
    cudaMalloc(&d_triangles, scene.numTriangles * sizeof(Triangle));
    cudaMemcpy(d_triangles, scene.triangles, scene.numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

    // -- Lights device --
    Light *d_lights;
    cudaMalloc(&d_lights, scene.numLights * sizeof(Light));
    cudaMemcpy(d_lights, scene.lights, scene.numLights * sizeof(Light), cudaMemcpyHostToDevice);

    // -- Camera device --
    Camera d_cam = {cam.position, cam.direction, cam.fov, cam.width, cam.height, Vecteur3D(1, 0, 0), Vecteur3D(0, 1, 0)};
    Camera *d_camera;
    cudaMalloc(&d_camera, sizeof(Camera));
    cudaMemcpy(d_camera, &d_cam, sizeof(Camera), cudaMemcpyHostToDevice);

    // -- Image --
    Color *d_image;
    cudaMalloc(&d_image, WIDTH_PIXEL * HEIGHT_PIXEL * sizeof(Color));

    // Mesure du temps kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float milliseconds = 0;
    float total_milliseconds = 0;

    auto t_start = high_resolution_clock::now();

    for (int i = 0; i < nb_images; i++)
    {
        sprintf(buffer, "video/frame%03d.ppm", i);

        // Cube rotation
        cube->rotateX((180.0 * nb_turns) / nb_images, cube->getCenter());
        cube->rotateY((360.0 * nb_turns) / nb_images, cube->getCenter());

        // Pyramid rotation
        /*pyramid->rotateX((180.0 * nb_turns) / nb_images, pyramid->getCenter());
        pyramid->rotateY((360.0 * nb_turns) / nb_images, pyramid->getCenter());*/

        // Sphere rotation
        /*sphere->rotateX((180.0 * nb_turns) / nb_images, sphere->getCenter());
        sphere->rotateY((360.0 * nb_turns) / nb_images, sphere->getCenter());*/

        // Copie des triangles mis à jour
        cudaMemcpy(d_triangles, cube->triangles, scene.numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

        // Image host
        Color *h_image = new Color[WIDTH_PIXEL * HEIGHT_PIXEL];

        // Kernel
        dim3 blockDim(128, 4);
        dim3 gridDim((WIDTH_PIXEL + blockDim.x - 1) / blockDim.x, (HEIGHT_PIXEL + blockDim.y - 1) / blockDim.y);

        cudaEventRecord(start);

        renderKernel<<<gridDim, blockDim>>>(d_image, d_triangles, scene.numTriangles, d_camera, d_lights, scene.numLights);

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);

        total_milliseconds += milliseconds;

        cudaDeviceSynchronize();

        // Copie vers host
        cudaMemcpy(h_image, d_image, WIDTH_PIXEL * HEIGHT_PIXEL * sizeof(Color), cudaMemcpyDeviceToHost);

        // Sauvegarde image
        savePPM(buffer, h_image);
        delete[] h_image;
    }

    auto t_end = high_resolution_clock::now();
    duration<double> t_total = t_end - t_start;

    // Nettoyage
    cudaFree(d_image);
    cudaFree(d_camera);
    cudaFree(d_lights);
    cudaFree(d_triangles);

    // Creation of the video using ffmpeg
    char ffmpegCommand[256];
    sprintf(ffmpegCommand, "ffmpeg -y -framerate %d -i ../build/video/frame%%03d.ppm -c:v libx264 -pix_fmt yuv420p output.mp4", fps);

    // Handle potential error
    ret = system(ffmpegCommand);
    if (ret != 0)
        cerr << "Erreur lors de la création de la vidéo avec ffmpeg." << endl;

    float fps2 = 1000.0f * nb_images / total_milliseconds;

    cout << "Generation kernel avec FPS moyen : " << fps2 << " (" << (total_milliseconds / 1000.0f) << "s)" << endl;
    cout << "Images générées avec FPS moyen : " << (nb_images / t_total.count()) << " (" << t_total.count() << "s)" << endl;

    return 0;
}
