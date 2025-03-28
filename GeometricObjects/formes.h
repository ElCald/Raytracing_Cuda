// formes.h
#ifndef FORMES_H
#define FORMES_H

#include "../Geometry/geometry.h"

class Color
{
public:
    int r, g, b;

    Color(int r = 0, int g = 0, int b = 0) : r(r), g(g), b(b) {}

    // Convertir la couleur en un entier représentant les 3 canaux RVB
    int toInt() const
    {
        return (r << 16) | (g << 8) | b;
    }
};

class Forme
{
public:
    Color couleur; // Ajouter une couleur à chaque forme

    Forme(Color _couleur) : couleur(_couleur) {}

    virtual bool intersection(const Ray &r, double &t) const = 0;
};

class Sphere : public Forme
{
public:
    Point3D centre;
    double rayon;

    Sphere(Point3D _centre, double _rayon, Color _couleur)
        : Forme(_couleur), centre(_centre), rayon(_rayon) {}

    bool intersection(const Ray &r, double &t) const override;
};

class Triangle : public Forme
{
public:
    Point3D p1, p2, p3;

    Triangle(Point3D _p1, Point3D _p2, Point3D _p3, Color _couleur)
        : Forme(_couleur), p1(_p1), p2(_p2), p3(_p3) {}

    bool intersection(const Ray &r, double &t) const override;
};

class Carre : public Forme
{
public:
    Point3D p1, p2, p3, p4;

    Carre(Point3D _p1, Point3D _p2, Point3D _p3, Point3D _p4, Color _couleur)
        : Forme(_couleur), p1(_p1), p2(_p2), p3(_p3), p4(_p4) {}

    bool intersection(const Ray &r, double &t) const override;
};

#endif // FORMES_H