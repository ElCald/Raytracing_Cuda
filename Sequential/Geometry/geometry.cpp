// Includes
#include "geometry.h"

// ---- Vecteur3D Implementation ----

/**
 * @param _x abscissa direction of the vector
 * @param _y ordinate direction of the vector
 * @param _z depth direction of the vector
 */
Vecteur3D::Vecteur3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

/**
 * @param v vector
 * @return scalar product
 */
double Vecteur3D::dot(const Vecteur3D &v) const
{
    return x * v.x + y * v.y + z * v.z;
}

/**
 * @return vector's norm
 */
double Vecteur3D::length() const
{
    return std::sqrt(x * x + y * y + z * z);
}

/**
 * @param v vector
 * @return vectorial vector product
 */
Vecteur3D Vecteur3D::cross(const Vecteur3D &v) const
{
    return Vecteur3D(
        y * v.z - z * v.y,
        z * v.x - x * v.z,
        x * v.y - y * v.x);
}

/**
 * @return unit vector
 */
Vecteur3D Vecteur3D::normalized() const
{
    double len = length();
    return (len > 0) ? Vecteur3D(x / len, y / len, z / len) : *this;
}

/**
 * @param v vector
 * @return addition of vectors
 */
Vecteur3D Vecteur3D::operator+(const Vecteur3D &v) const
{
    return Vecteur3D(x + v.x, y + v.y, z + v.z);
}

/**
 * @param v vector
 * @return substraction of vectors
 */
Vecteur3D Vecteur3D::operator-(const Vecteur3D &v) const
{
    return Vecteur3D(x - v.x, y - v.y, z - v.z);
}

/**
 * @param scalar scalar
 * @return multiplication of vector
 */
Vecteur3D Vecteur3D::operator*(double scalar) const
{
    return Vecteur3D(x * scalar, y * scalar, z * scalar);
}

/**
 * @param scalar scalar
 * @return division of vector
 */
Vecteur3D Vecteur3D::operator/(double scalar) const
{
    return Vecteur3D(x / scalar, y / scalar, z / scalar);
}

/**
 * @param v vector
 * @return multiplication of vector
 */
Vecteur3D Vecteur3D::operator*(const Vecteur3D &v) const
{
    return Vecteur3D(x * v.x, y * v.y, z * v.z);
}

/**
 * @param v vector
 * @return addition of vector
 */
Vecteur3D &Vecteur3D::operator+=(const Vecteur3D &v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

/**
 * @return inverse vector
 */
Vecteur3D Vecteur3D::inverse() const
{
    return Vecteur3D(-x, -y, -z);
}

// ---- Point3D implementation ----

/**
 * @param _x abscissa coordinate of the point
 * @param _y ordinate coordinate of the point
 * @param _z depth coordinate of the point
 */
Point3D::Point3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

/**
 * @param p point
 * @return substraction of points
 */
Vecteur3D Point3D::operator-(const Point3D &p) const
{
    return Vecteur3D(x - p.x, y - p.y, z - p.z);
}

/**
 * @param p point
 * @return addition of points
 */
Point3D Point3D::operator+(const Point3D &p) const
{
    return Point3D(x + p.x, y + p.y, z + p.z);
}

/**
 * @param v vector
 * @return addition of point and vector
 */
Point3D Point3D::operator+(const Vecteur3D &v) const
{
    return Point3D(x + v.x, y + v.y, z + v.z);
}

// ---- Ray implementation ----

/**
 * @param orig origin of the ray
 * @param direc direction of the ray
 */
Ray::Ray(const Point3D &orig, const Vecteur3D &direc)
    : origine(orig), direction(direc.normalized()) {}

/**
 * @param t vector
 * @return distance from the origin on the ray
 */
Point3D Ray::at(double t) const
{
    return Point3D(
        origine.x + t * direction.x,
        origine.y + t * direction.y,
        origine.z + t * direction.z);
}
