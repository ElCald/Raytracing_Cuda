#include "scene.h"

Scene::Scene() {}
Scene::~Scene() {}

void Scene::addObject(Forme *obj)
{
    objets.push_back(obj);
}

bool Scene::intersect(const Ray &ray, Point3D &hitPoint, Vecteur3D &normal, int &materiau) const
{
    bool hit = false;
    double minDist = 1e9;
    for (const auto &obj : objets)
    {
        if (obj->intersection(ray.origine))
        { // Simplification, améliorer la détection
            hit = true;
            materiau = obj->materiau;
        }
    }
    return hit;
}