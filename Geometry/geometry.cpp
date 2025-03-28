#include "geometry.h"

// ---- Implémentation de Point3D ----
Point3D::Point3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

// Point3D Point3D::operator-(const Point3D &p) const
// {
//     return Point3D(x - p.x, y - p.y, z - p.z);
// }


Vecteur3D Point3D::operator-(const Point3D &p) const
{
    return Vecteur3D(x - p.x, y - p.y, z - p.z);
}



Point3D Point3D::operator+(const Point3D &p) const
{
    return Point3D(x + p.x, y + p.y, z + p.z);
}


Point3D Point3D::operator+(const Vecteur3D& v) const {
    return Point3D(x + v.x, y + v.y, z + v.z);
}



// ---- Implémentation de Vecteur3D ----
Vecteur3D::Vecteur3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

double Vecteur3D::dot(const Vecteur3D &v) const
{
    return x * v.x + y * v.y + z * v.z;
}

Vecteur3D Vecteur3D::cross(const Vecteur3D &v) const
{
    return Vecteur3D(
        y * v.z - z * v.y,
        z * v.x - x * v.z,
        x * v.y - y * v.x);
}

double Vecteur3D::length() const
{
    return std::sqrt(x * x + y * y + z * z);
}

Vecteur3D Vecteur3D::normalized() const
{
    double len = length();
    return (len > 0) ? Vecteur3D(x / len, y / len, z / len) : *this;
}

// ---- Opérateurs de Vecteur3D ----
Vecteur3D Vecteur3D::operator+(const Vecteur3D &v) const
{
    return Vecteur3D(x + v.x, y + v.y, z + v.z);
}

Vecteur3D Vecteur3D::operator-(const Vecteur3D &v) const
{
    return Vecteur3D(x - v.x, y - v.y, z - v.z);
}

Vecteur3D Vecteur3D::operator*(double scalar) const
{
    return Vecteur3D(x * scalar, y * scalar, z * scalar);
}

Vecteur3D Vecteur3D::operator/(double scalar) const
{
    return Vecteur3D(x / scalar, y / scalar, z / scalar);
}

// ---- Implémentation de Ray ----
Ray::Ray(const Point3D &orig, const Vecteur3D &direc)
    : origine(orig), direction(direc.normalized()) {}

Point3D Ray::at(double t) const
{
    return Point3D(
        origine.x + t * direction.x,
        origine.y + t * direction.y,
        origine.z + t * direction.z);
}
