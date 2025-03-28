#include "formes.h"
#include <cmath>

// Intersection d'un rayon avec une sphère
bool Sphere::intersection(const Ray &r, double &t) const
{
    // Le rayon est donné par : r(t) = origine + t * direction
    // On substitue r(t) dans l'équation de la sphère

    // Calcul de l'élément de produit scalaire
    Vecteur3D oc = r.origine - centre;
    double a = r.direction.dot(r.direction);
    double b = 2.0 * oc.dot(r.direction);
    double c = oc.dot(oc) - rayon * rayon;

    // Calcul du discriminant
    double discriminant = b * b - 4 * a * c;

    if (discriminant > 0)
    {
        // Il y a deux intersections possibles
        double t0 = (-b - sqrt(discriminant)) / (2.0 * a);
        double t1 = (-b + sqrt(discriminant)) / (2.0 * a);

        // On choisit le plus proche point d'intersection (t > 0)
        if (t0 > 0 && t1 > 0)
        {
            t = (t0 < t1) ? t0 : t1;
            return true;
        }
        else if (t0 > 0)
        {
            t = t0;
            return true;
        }
        else if (t1 > 0)
        {
            t = t1;
            return true;
        }
    }
    return false;
}

// Intersection d'un rayon avec un triangle
bool Triangle::intersection(const Ray &r, double &t) const
{
    // Vecteurs pour les bords du triangle
    Vecteur3D e1 = p2 - p1;
    Vecteur3D e2 = p3 - p1;

    // Calcul du produit vectoriel entre la direction du rayon et l'un des bords
    Vecteur3D h = r.direction.cross(e2);
    double a = e1.dot(h);

    // Si le produit scalaire est proche de zéro, il n'y a pas d'intersection
    if (a > -1e-6 && a < 1e-6)
        return false;

    double f = 1.0 / a;
    Vecteur3D s = r.origine - p1;
    double u = f * s.dot(h);

    if (u < 0.0 || u > 1.0)
        return false;

    Vecteur3D q = s.cross(e1);
    double v = f * r.direction.dot(q);

    if (v < 0.0 || u + v > 1.0)
        return false;

    // Calcul de t, la distance de l'intersection
    t = f * e2.dot(q);

    if (t > 1e-6)
        return true;
    else
        return false;
}

// Intersection d'un rayon avec un carré (simplifié comme 4 bords)
bool Carre::intersection(const Ray &r, double &t) const
{
    // Vous devez vérifier l'intersection avec chaque côté du carré.
    // Par exemple, en décomposant le carré en 2 triangles et en vérifiant
    // si le rayon intersecte l'un ou l'autre.

    // Exemple avec une méthode d'intersection avec des triangles
    // Divisez le carré en 2 triangles et vérifiez chaque triangle
    Triangle triangle1(p1, p2, p3, Color(0, 0, 0)); // Premier triangle du carré
    Triangle triangle2(p1, p3, p4, Color(0, 0, 0)); // Deuxième triangle du carré

    double t1, t2;
    if (triangle1.intersection(r, t1) || triangle2.intersection(r, t2))
    {
        t = std::min(t1, t2); // Choisir le plus proche
        return true;
    }
    return false;
}