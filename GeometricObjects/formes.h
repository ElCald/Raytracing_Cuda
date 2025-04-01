// formes.h
#ifndef FORMES_H
#define FORMES_H

// Includes
#include "../Geometry/geometry.h"
#include <vector>

using namespace std;

// Material class
class Material
{
public:
    Material(Vecteur3D _ambient, Vecteur3D _diffuse, Vecteur3D _specular, float _shininess);
    Vecteur3D ambient;
    Vecteur3D diffuse;
    Vecteur3D specular;
    float shininess;
};

// Color class
class Color
{
public:
    int r, g, b;

    Color(int r, int g, int b);

    int toInt() const;
};

// Forme class
class Forme
{
public:
    Material materiau;

    Forme(Material _materiau);

    virtual bool intersection(const Ray &r, double &t) const = 0;
    virtual Vecteur3D getNormal(const Point3D &p) const = 0;
};

// Sphere class
class Sphere : public Forme
{
public:
    Point3D centre;
    double rayon;

    Sphere(Point3D _centre, double _rayon, Material _materiau);

    bool intersection(const Ray &r, double &t) const override;
    Vecteur3D getNormal(const Point3D &p) const override;
};

// Triangle class
class Triangle : public Forme
{
public:
    Point3D p1, p2, p3;

    Triangle(Point3D _p1, Point3D _p2, Point3D _p3, Material _materiau);

    bool intersection(const Ray &r, double &t) const override;
    Vecteur3D getNormal(const Point3D &p) const override;
    bool contains(const Point3D &p) const;
};

// Carre class
class Carre : public Forme
{
public:
    Point3D p1, p2, p3, p4;

    Carre(Point3D _p1, Point3D _p2, Point3D _p3, Point3D _p4, Material _materiau);

    bool intersection(const Ray &r, double &t) const override;
    Vecteur3D getNormal(const Point3D &p) const override;
};

// Cube class
class Cube : public Forme
{
private:
    double size;
    Point3D center;

public:
    vector<Triangle *> cube;

    Cube(double _size, const Point3D &_center, Material _materiau);

    void rotateX(double angle);
    void rotateY(double angle);
    void rotateZ(double angle);

    void translateX(double direc);
    void translateY(double direc);
    void translateZ(double direc);

    bool intersection(const Ray &r, double &t) const override;
    Vecteur3D getNormal(const Point3D &p) const override;
};

Point3D rotateAroundX(const Point3D &P, const Point3D &O, double angle);
Point3D rotateAroundY(const Point3D &P, const Point3D &O, double angle);
Point3D rotateAroundZ(const Point3D &P, const Point3D &O, double angle);

#endif // FORMES_H