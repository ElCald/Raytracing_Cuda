// Includes
#include "forms.h"
#include <cmath>

using namespace std;

// ---- Material Implementation ----

/**
 * @param _ambient ambient light
 * @param _diffuse diffuse light
 * @param _specular specular light
 * @param _shininess shining of the light
 */
__host__ __device__ Material::Material(Vecteur3D _ambient, Vecteur3D _diffuse, Vecteur3D _specular, float _shininess) : ambient(_ambient), diffuse(_diffuse), specular(_specular), shininess(_shininess) {}

// ---- Color Implementation ----

/**
 * default constructor
 */
__host__ __device__ Color::Color() : r(0), g(0), b(0) {}

/**
 * @param r red
 * @param  g green
 * @param b blue
 */
__host__ __device__ Color::Color(int _r = 0, int _g = 0, int _b = 0) : r(_r), g(_g), b(_b) {}

/**
 * @return to int of RGB for ppm image
 */
__host__ __device__ int Color::toInt() const
{
    return (r << 16) | (g << 8) | b;
}

// ---- Form Implementation ----

/**
 * @param _material material
 */
__host__ __device__ Form::Form(Material _material) : materiau(_material) {}

// ---- Sphere Implementation ----

/**
 * @param _center center of the sphere
 * @param _rayon sphere's radius
 * @param _material sphere's material
 */
__host__ __device__ Sphere::Sphere(Point3D _center, double _rayon, Material _material) : Form(_material), center(_center), rayon(_rayon) {}

/**
 * @param r ray
 * @param t point on the ray
 * @return true if there is intersection between ray and sphere
 */
__host__ __device__ bool Sphere::intersection(const Ray &r, double &t) const
{
    // Ray given by : r(t) = origin + t * direction
    // Substitute r(t) into the equation for the sphere

    // Calculating the scalar product element
    Vecteur3D oc = r.origine - center;
    double a = r.direction.dot(r.direction);
    double b = 2.0 * oc.dot(r.direction);
    double c = oc.dot(oc) - rayon * rayon;

    // Calculating the discriminant
    double discriminant = b * b - 4 * a * c;

    if (discriminant > 0)
    {
        // There are two possible intersections
        double t0 = (-b - sqrt(discriminant)) / (2.0 * a);
        double t1 = (-b + sqrt(discriminant)) / (2.0 * a);

        // Choose the nearest point of intersection (t > 0)
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

/**
 * @param p point
 * @return the normal of the sphere
 */
__host__ __device__ Vecteur3D Sphere::getNormal(const Point3D &p) const
{
    return (p - this->center).normalized();
}

// ---- Triangle Implementation ----

/**
 * @param _p1 first point of the triangle
 * @param _p2 second point of the triangle
 * @param _p3 third point of the triangle
 * @param _material material
 */
__host__ __device__ Triangle::Triangle(Point3D _p1, Point3D _p2, Point3D _p3, Material _material) : Form(_material), p1(_p1), p2(_p2), p3(_p3) {}

/**
 * @param r ray
 * @param t point on the ray
 * @return true if there is intersection between ray and triangle
 */
__host__ __device__ bool Triangle::intersection(const Ray &r, double &t) const
{
    // Vectors for the edges of the triangle
    Vecteur3D e1 = p2 - p1;
    Vecteur3D e2 = p3 - p1;

    // Calculation of the vector product between the direction of the ray and one of the edges
    Vecteur3D h = r.direction.cross(e2);
    double a = e1.dot(h);

    // If the scalar product is close to zero, there is no intersection
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

    // Calculate t, the distance of the intersection
    t = f * e2.dot(q);

    if (t > 1e-6)
        return true;
    else
        return false;
}

/**
 * @param p point
 * @return true if the point is on the triangle
 */
__host__ __device__ bool Triangle::contains(const Point3D &p) const
{
    // Vectors formed by the sides of the triangle
    Vecteur3D v0 = p2 - p1;
    Vecteur3D v1 = p3 - p1;
    Vecteur3D v2 = p - p1;

    // Calculating scalar products
    double dot00 = v0.dot(v0);
    double dot01 = v0.dot(v1);
    double dot02 = v0.dot(v2);
    double dot11 = v1.dot(v1);
    double dot12 = v1.dot(v2);

    // Calculation of the denominator (invariant)
    double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);

    // Calculating barycentrics
    double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    // The point is inside if u >= 0, v >= 0, and u + v <= 1
    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

/**
 * @param p point
 * @return the normal of the triangle
 */
__host__ __device__ Vecteur3D Triangle::getNormal([[maybe_unused]] const Point3D &p) const
{
    // Vectors AB and AC
    Vecteur3D AB = p2 - p1;
    Vecteur3D AC = p3 - p1;

    // The vector product between AB and AC gives the normal
    return AB.cross(AC).normalized();
}

// ---- Square Implementation ----

/**
 * @param _p1 first point of the triangle
 * @param _p2 second point of the triangle
 * @param _p3 third point of the triangle
 * @param _p4 fourth point of the triangle
 * @param _material material
 */
__host__ __device__ Square::Square(Point3D _p1, Point3D _p2, Point3D _p3, Point3D _p4, Material _material) : Form(_material), p1(_p1), p2(_p2), p3(_p3), p4(_p4) {}

/**
 * @param r ray
 * @param t point on the ray
 * @return true if there is intersection between ray and square
 */
__host__ __device__ bool Square::intersection(const Ray &r, double &t) const
{
    // Square is divided into 2 triangles
    Triangle triangle1(p1, p2, p3, {Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0, 0, 0), Vecteur3D(1, 1, 1), 32}); // Premier triangle du carré
    Triangle triangle2(p1, p3, p4, {Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0, 0, 0), Vecteur3D(1, 1, 1), 32}); // Deuxième triangle du carré

    double t1, t2;
    if (triangle1.intersection(r, t1) || triangle2.intersection(r, t2))
    {
        t = std::min(t1, t2); // Choose the closest one
        return true;
    }
    return false;
}

/**
 * @param p point
 * @return the normal of the square
 */
__host__ __device__ Vecteur3D Square::getNormal([[maybe_unused]] const Point3D &p) const
{
    // We take the first three points to define the plan
    Vecteur3D AB = p2 - p1;
    Vecteur3D AD = p4 - p1;

    // The vector product between AB and AD gives the normal
    return AB.cross(AD).normalized();
}

// ---- Cube Implementation ----

/**
 * @param _size size of the cube
 * @param _center center of the cube
 * @param _material material
 */
__host__ __device__ Cube::Cube(double _size, const Point3D &_center, Material _material) : Form(_material), size(_size), center(_center)
{

    // front face
    Triangle *t1 = new Triangle(Point3D(0, 0, 0), Point3D(_size, 0, 0), Point3D(_size, _size, 0), _material);
    cube.push_back(t1);

    Triangle *t2 = new Triangle(Point3D(_size, _size, 0), Point3D(0, _size, 0), Point3D(0, 0, 0), _material);
    cube.push_back(t2);

    // upper face
    Triangle *t3 = new Triangle(Point3D(0, _size, 0), Point3D(_size, _size, 0), Point3D(_size, _size, -_size), _material);
    cube.push_back(t3);

    Triangle *t4 = new Triangle(Point3D(_size, _size, -_size), Point3D(0, _size, -_size), Point3D(0, _size, 0), _material);
    cube.push_back(t4);

    // right face
    Triangle *t5 = new Triangle(Point3D(_size, 0, 0), Point3D(_size, 0, -_size), Point3D(_size, _size, -_size), _material);
    cube.push_back(t5);

    Triangle *t6 = new Triangle(Point3D(_size, _size, -_size), Point3D(_size, _size, 0), Point3D(_size, 0, 0), _material);
    cube.push_back(t6);

    // left face
    Triangle *t7 = new Triangle(Point3D(0, 0, -_size), Point3D(0, 0, 0), Point3D(0, _size, 0), _material);
    cube.push_back(t7);

    Triangle *t8 = new Triangle(Point3D(0, _size, 0), Point3D(0, _size, -_size), Point3D(0, 0, -_size), _material);
    cube.push_back(t8);

    // lower face
    Triangle *t9 = new Triangle(Point3D(0, 0, -_size), Point3D(_size, 0, -_size), Point3D(_size, 0, 0), _material);
    cube.push_back(t9);

    Triangle *t10 = new Triangle(Point3D(_size, 0, 0), Point3D(0, 0, 0), Point3D(0, 0, -_size), _material);
    cube.push_back(t10);

    // rear face
    Triangle *t11 = new Triangle(Point3D(0, _size, -_size), Point3D(_size, _size, -_size), Point3D(_size, 0, -_size), _material);
    cube.push_back(t11);

    Triangle *t12 = new Triangle(Point3D(_size, 0, -_size), Point3D(0, 0, -_size), Point3D(0, _size, -_size), _material);
    cube.push_back(t12);

    int x_temp = center.x, y_temp = center.y, z_temp = center.z;

    translateX(center.x);
    translateY(center.y);
    translateZ(center.z);

    center.x = x_temp;
    center.y = y_temp;
    center.z = z_temp;
}

/**
 * @return center of the cube
 */
__host__ __device__ Point3D Cube::getCenter()
{
    return center;
}

/**
 * @return size of the cube
 */
__host__ __device__ double Cube::getSize()
{
    return size;
}

/**
 * @param angle angle of the rotation
 * @param center center of the cube
 */
__host__ __device__ void Cube::rotateX(double angle, Point3D center)
{
    for (auto t : cube)
    {
        t->p1 = rotateAroundX(t->p1, center, angle);
        t->p2 = rotateAroundX(t->p2, center, angle);
        t->p3 = rotateAroundX(t->p3, center, angle);
    }
}

/**
 * @param angle angle of the rotation
 * @param center center of the cube
 */
__host__ __device__ void Cube::rotateY(double angle, Point3D center)
{
    for (auto t : cube)
    {
        t->p1 = rotateAroundY(t->p1, center, angle);
        t->p2 = rotateAroundY(t->p2, center, angle);
        t->p3 = rotateAroundY(t->p3, center, angle);
    }
}

/**
 * @param angle angle of the rotation
 * @param center center of the cube
 */
__host__ __device__ void Cube::rotateZ(double angle, Point3D center)
{
    for (auto t : cube)
    {
        t->p1 = rotateAroundZ(t->p1, center, angle);
        t->p2 = rotateAroundZ(t->p2, center, angle);
        t->p3 = rotateAroundZ(t->p3, center, angle);
    }
}

/**
 * @param direc size of the translation
 */
__host__ __device__ void Cube::translateX(double direc)
{
    for (auto t : cube)
    {
        t->p1.x += direc;
        t->p2.x += direc;
        t->p3.x += direc;
    }

    center.x += direc;
}

/**
 * @param direc size of the translation
 */
__host__ __device__ void Cube::translateY(double direc)
{
    for (auto t : cube)
    {
        t->p1.y += direc;
        t->p2.y += direc;
        t->p3.y += direc;
    }

    center.y += direc;
}

/**
 * @param direc size of the translation
 */
__host__ __device__ void Cube::translateZ(double direc)
{
    for (auto t : cube)
    {
        t->p1.z += direc;
        t->p2.z += direc;
        t->p3.z += direc;
    }

    center.z += direc;
}

/**
 * @param r ray
 * @param t point on the ray
 * @return true if there is intersection between ray and square
 * Default implementation
 */
__host__ __device__ bool Cube::intersection([[maybe_unused]] const Ray &r, [[maybe_unused]] double &t) const
{
    return false;
}

/**
 * @param p point
 * @return the normal of the cube
 * * Default implementation
 */
__host__ __device__ Vecteur3D Cube::getNormal([[maybe_unused]] const Point3D &p) const
{
    return Vecteur3D();
}

// ---- Other functions ----

/**
 * @param P first point
 * @param O second point
 * @param angle angle of the rotation
 * @return new point
 */
__host__ __device__ Point3D rotateAroundX(const Point3D &P, const Point3D &O, double angle)
{
    // Convert angle to radians
    double rad = angle * M_PI / 180.0;
    double cosAngle = cos(rad);
    double sinAngle = sin(rad);

    // Translate point P so that O is at the origin
    double dy = P.y - O.y;
    double dz = P.z - O.z;

    // Apply the rotation matrix around the X axis
    double newY = cosAngle * dy - sinAngle * dz;
    double newZ = sinAngle * dy + cosAngle * dz;

    // Return to original position
    return Point3D(P.x, newY + O.y, newZ + O.z);
}

/**
 * @param P first point
 * @param O second point
 * @param angle angle of the rotation
 * @return new point
 */
__host__ __device__ Point3D rotateAroundY(const Point3D &P, const Point3D &O, double angle)
{
    double rad = angle * M_PI / 180.0;
    double s = sin(rad);
    double c = cos(rad);

    // Translation to bring the centre back to (0,0,0)
    double x = P.x - O.x;
    double z = P.z - O.z;

    // Rotate around Y
    double new_x = x * c + z * s;
    double new_z = -x * s + z * c;

    // Re-translation
    return Point3D(new_x + O.x, P.y, new_z + O.z);
}

/**
 * @param P first point
 * @param O second point
 * @param angle angle of the rotation
 * @return new point
 */
__host__ __device__ Point3D rotateAroundZ(const Point3D &P, const Point3D &O, double angle)
{
    // Convert angle to radians
    double rad = angle * M_PI / 180.0;
    double cosAngle = cos(rad);
    double sinAngle = sin(rad);

    // Translate point P so that O is at the origin
    double dx = P.x - O.x;
    double dy = P.y - O.y;

    // Apply the rotation matrix around the Z axis
    double newX = cosAngle * dx - sinAngle * dy;
    double newY = sinAngle * dx + cosAngle * dy;

    // Return to original position
    return Point3D(newX + O.x, newY + O.y, P.z);
}
