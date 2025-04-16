// scene.h
#ifndef SCENE_H
#define SCENE_H

// Includes
#include "../GeometricsObjects/forms.h"
#include "light.h"
#include "camera.h"
#include <vector>

using namespace std;

// Scene class
class Scene
{
public:
    // Attributes
    Camera &camera;
    vector<Light *> lights;

    // Constructor and Deletor
    Scene(Camera &_camera);
    ~Scene();

    // Methods
    void addObject(vector<Triangle *> &obj);
    void addObject(Form *obj);
    void addLight(Light *light);
    void render(vector<vector<Color>> &image, int width, int height);

private:
    // Attribute
    vector<Form *> objects;
};

// Phong function
Vecteur3D phongShading(const Point3D &point, const Vecteur3D &normal, const Vecteur3D &viewDir, const std::vector<Light *> &lights, const Material &material);

#endif // SCENE_H