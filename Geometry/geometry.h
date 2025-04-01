#ifndef GEOMETRY_H
#define GEOMETRY_H

// Includes
#include <cmath>

// 3D Vector class
class Vecteur3D
{
public:
    double x, y, z;

    Vecteur3D(double _x = 0, double _y = 0, double _z = 0);

    double dot(const Vecteur3D &v) const;
    double length() const;
    Vecteur3D cross(const Vecteur3D &v) const;
    Vecteur3D normalized() const;

    Vecteur3D operator+(const Vecteur3D &v) const;
    Vecteur3D operator-(const Vecteur3D &v) const;
    Vecteur3D operator*(double scalar) const;
    Vecteur3D operator/(double scalar) const;
    Vecteur3D operator*(const Vecteur3D &v) const;
    Vecteur3D inverse() const;
};

// 3D Point class
class Point3D
{
public:
    double x, y, z;

    Point3D(double _x = 0, double _y = 0, double _z = 0);

    Vecteur3D operator-(const Point3D &p) const;
    Point3D operator+(const Point3D &p) const;
    Point3D operator+(const Vecteur3D &v) const;
};

// Ray class
class Ray
{
public:
    Point3D origine;
    Vecteur3D direction;

    Ray(const Point3D &orig, const Vecteur3D &direc);
    Point3D at(double t) const;
};

#endif // GEOMETRY_H
