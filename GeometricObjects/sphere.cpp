#include "formes.h"

#include <cmath>

using namespace std;


Sphere::Sphere() {}

Sphere::Sphere(Point3D _centre, double _rayon) : centre(_centre), rayon(_rayon) {}



bool Sphere::intersection(const Ray& ray, double& t) const {
    Vecteur3D oc = Vecteur3D(ray.origine.x - centre.x, 
                             ray.origine.y - centre.y, 
                             ray.origine.z - centre.z);

    double a = ray.direction.dot(ray.direction);
    double b = 2.0 * oc.dot(ray.direction);
    double c = oc.dot(oc) - rayon * rayon;
    double discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        return false;  // Pas d'intersection
    } else {
        double sqrtD = sqrt(discriminant);
        double t1 = (-b - sqrtD) / (2.0 * a);  // Première solution
        double t2 = (-b + sqrtD) / (2.0 * a);  // Deuxième solution

        // Choisir la plus proche intersection positive
        if (t1 > 0) {
            t = t1;
            return true;
        }
        if (t2 > 0) {
            t = t2;
            return true;
        }
        return false;
    }
}