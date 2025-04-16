// geometry.h
#ifndef GEOMETRY_H
#define GEOMETRY_H

// Includes
#include <cmath>

// 3D Vector class
class Vecteur3D
{
public:
    // Attributes
    double x, y, z;

    // Constructor
    Vecteur3D(double _x = 0, double _y = 0, double _z = 0);

    // Methods
    double dot(const Vecteur3D &v) const;
    double length() const;
    Vecteur3D cross(const Vecteur3D &v) const;
    Vecteur3D normalized() const;
    Vecteur3D inverse() const;

    // Operators
    Vecteur3D operator+(const Vecteur3D &v) const;
    Vecteur3D operator-(const Vecteur3D &v) const;
    Vecteur3D operator*(double scalar) const;
    Vecteur3D operator/(double scalar) const;
    Vecteur3D operator*(const Vecteur3D &v) const;
    Vecteur3D &operator+=(const Vecteur3D &v);
};

// 3D Point class
class Point3D
{
public:
    // Attributes
    double x, y, z;

    // Constructor
    Point3D(double _x = 0, double _y = 0, double _z = 0);

    // Operators
    Vecteur3D operator-(const Point3D &p) const;
    Point3D operator+(const Point3D &p) const;
    Point3D operator+(const Vecteur3D &v) const;
};

// Ray class
class Ray
{
public:
    // Attributes
    Point3D origine;
    Vecteur3D direction;

    // Constructor
    Ray(const Point3D &orig, const Vecteur3D &direc);

    // Method
    Point3D at(double t) const;
};

#endif // GEOMETRY_H
