#ifndef SCENE_H
#define SCENE_H

#include "../GeometricObjects/formes.h"
#include <vector>

class Scene
{
public:
    Scene();
    ~Scene();

    void addObject(Forme *obj);
    bool intersect(const Ray &ray, Point3D &hitPoint, Vecteur3D &normal, int &materiau) const;

private:
    std::vector<Forme *> objets;
};

#endif // SCENE_H