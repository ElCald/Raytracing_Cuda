#ifndef SCENE_H
#define SCENE_H

#include "../GeometricObjects/formes.h"
#include "light.h"
#include "camera.h"
#include <vector>

class Scene
{
public:
    Camera &camera;
    std::vector<Light *> lights;

    Scene(Camera &_camera);
    ~Scene();

    void addObject(vector<Triangle *> &obj);
    void addObject(Forme *obj);
    void addLight(Light *light);

    // bool intersect(const Ray &ray, Point3D &hitPoint, Vecteur3D &normal, int &materiau) const;

    void render(vector<vector<Color>> &image, int width, int height);

private:
    std::vector<Forme *> objects;
};

Vecteur3D phongShading(const Point3D &point, const Vecteur3D &normal, const Vecteur3D &viewDir, const std::vector<Light *> lights, const Material &material);

#endif // SCENE_H