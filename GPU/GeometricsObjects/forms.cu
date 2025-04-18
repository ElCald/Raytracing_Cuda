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

// ---- Triangle Implementation ----

/**
 * @param _p1 first point of the triangle
 * @param _p2 second point of the triangle
 * @param _p3 third point of the triangle
 * @param _material material
 */
__host__ __device__ Triangle::Triangle(Point3D _p1, Point3D _p2, Point3D _p3, Material _material) : mat(_material), p1(_p1), p2(_p2), p3(_p3) {}

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

// ---- Cube Implementation ----

/**
 * @brief Constructor for the cube.
 * @param _size Size of the cube.
 * @param _center Center of the cube.
 * @param _mat Material of the cube.
 */
__host__ __device__ Cube::Cube(double _size, const Point3D &_center, const Material &_mat)
    : size(_size), center(_center)
{
    Point3D A = Point3D(0, 0, 0);
    Point3D B = Point3D(_size, 0, 0);
    Point3D C = Point3D(_size, _size, 0);
    Point3D D = Point3D(0, _size, 0);
    Point3D E = Point3D(0, 0, -_size);
    Point3D F = Point3D(_size, 0, -_size);
    Point3D G = Point3D(_size, _size, -_size);
    Point3D H = Point3D(0, _size, -_size);

    // Front face
    triangles[0] = Triangle(A, B, C, _mat);
    triangles[1] = Triangle(C, D, A, _mat);
    // Back face
    triangles[2] = Triangle(H, G, F, _mat);
    triangles[3] = Triangle(F, E, H, _mat);
    // Top face
    triangles[4] = Triangle(D, C, G, _mat);
    triangles[5] = Triangle(G, H, D, _mat);
    // Bottom face
    triangles[6] = Triangle(A, E, F, _mat);
    triangles[7] = Triangle(F, B, A, _mat);
    // Left face
    triangles[8] = Triangle(A, D, H, _mat);
    triangles[9] = Triangle(H, E, A, _mat);
    // Right face
    triangles[10] = Triangle(B, F, G, _mat);
    triangles[11] = Triangle(G, C, B, _mat);

    translateX(center.x);
    translateY(center.y);
    translateZ(center.z);
}

/**
 * @brief Rotates the cube around the X axis.
 * @param angle Rotation angle in degrees.
 * @param c Center of rotation.
 */
__host__ __device__ void Cube::rotateX(double angle, const Point3D &c)
{
    for (int i = 0; i < 12; ++i)
    {
        triangles[i].p1 = rotateAroundX(triangles[i].p1, c, angle);
        triangles[i].p2 = rotateAroundX(triangles[i].p2, c, angle);
        triangles[i].p3 = rotateAroundX(triangles[i].p3, c, angle);
    }
}

/**
 * @brief Rotates the cube around the Y axis.
 * @param angle Rotation angle in degrees.
 * @param c Center of rotation.
 */
__host__ __device__ void Cube::rotateY(double angle, const Point3D &c)
{
    for (int i = 0; i < 12; ++i)
    {
        triangles[i].p1 = rotateAroundY(triangles[i].p1, c, angle);
        triangles[i].p2 = rotateAroundY(triangles[i].p2, c, angle);
        triangles[i].p3 = rotateAroundY(triangles[i].p3, c, angle);
    }
}

/**
 * @brief Rotates the cube around the Z axis.
 * @param angle Rotation angle in degrees.
 * @param c Center of rotation.
 */
__host__ __device__ void Cube::rotateZ(double angle, const Point3D &c)
{
    for (int i = 0; i < 12; ++i)
    {
        triangles[i].p1 = rotateAroundZ(triangles[i].p1, c, angle);
        triangles[i].p2 = rotateAroundZ(triangles[i].p2, c, angle);
        triangles[i].p3 = rotateAroundZ(triangles[i].p3, c, angle);
    }
}

/**
 * @brief Translates the cube along the X axis.
 * @param val Translation value.
 */
__host__ __device__ void Cube::translateX(double val)
{
    for (int i = 0; i < 12; ++i)
    {
        triangles[i].p1.x += val;
        triangles[i].p2.x += val;
        triangles[i].p3.x += val;
    }
    center.x += val;
}

/**
 * @brief Translates the cube along the Y axis.
 * @param val Translation value.
 */
__host__ __device__ void Cube::translateY(double val)
{
    for (int i = 0; i < 12; ++i)
    {
        triangles[i].p1.y += val;
        triangles[i].p2.y += val;
        triangles[i].p3.y += val;
    }
    center.y += val;
}

/**
 * @brief Translates the cube along the Z axis.
 * @param val Translation value.
 */
__host__ __device__ void Cube::translateZ(double val)
{
    for (int i = 0; i < 12; ++i)
    {
        triangles[i].p1.z += val;
        triangles[i].p2.z += val;
        triangles[i].p3.z += val;
    }
    center.z += val;
}

/**
 * @brief Gets the center of the cube.
 * @return The center point of the cube.
 */
__host__ __device__ Point3D Cube::getCenter() const
{
    return center;
}

/**
 * @brief Gets the size of the cube.
 * @return The size (edge length) of the cube.
 */
__host__ __device__ double Cube::getSize() const
{
    return size;
}

// ---- Pyramid Implementation ----

/**
 * @brief Constructor for the pyramid.
 * @param b1 First point of the base pyramid.
 * @param b2 Second point of the base pyramid.
 * @param b3 Third point of the base pyramid.
 * @param apex Apex of the pyramid.
 * @param mat Material of the pyramid.
 * @param _center Center of the pyramid.
 */
__host__ __device__ Pyramid::Pyramid(const Point3D &b1, const Point3D &b2, const Point3D &b3, const Point3D &apex, const Material &mat, const Point3D &_center) : center(center)
{
    triangles[0] = Triangle(b1, b2, b3, mat);   // Base
    triangles[1] = Triangle(b1, b2, apex, mat); // Side 1
    triangles[2] = Triangle(b2, b3, apex, mat); // Side 2
    triangles[3] = Triangle(b3, b1, apex, mat); // Side 3
}

/**
 * @brief Rotates the pyramid around the X axis.
 * @param angle Rotation angle in degrees.
 * @param c Center of rotation.
 */
__host__ __device__ void Pyramid::rotateX(double angle, const Point3D &c)
{
    for (int i = 0; i < 4; ++i)
    {
        triangles[i].p1 = rotateAroundX(triangles[i].p1, c, angle);
        triangles[i].p2 = rotateAroundX(triangles[i].p2, c, angle);
        triangles[i].p3 = rotateAroundX(triangles[i].p3, c, angle);
    }
}

/**
 * @brief Rotates the pyramid around the Y axis.
 * @param angle Rotation angle in degrees.
 * @param c Center of rotation.
 */
__host__ __device__ void Pyramid::rotateY(double angle, const Point3D &c)
{
    for (int i = 0; i < 4; ++i)
    {
        triangles[i].p1 = rotateAroundY(triangles[i].p1, c, angle);
        triangles[i].p2 = rotateAroundY(triangles[i].p2, c, angle);
        triangles[i].p3 = rotateAroundY(triangles[i].p3, c, angle);
    }
}

/**
 * @brief Gets the center of the pyramid.
 * @return The center point of the pyramid.
 */
__host__ __device__ Point3D Pyramid::getCenter() const
{
    return center;
}

// ---- Sphere Generation ----

/**
 * @brief Constructor for the triangle-based sphere.
 * @param _center Center of the sphere.
 */
__host__ TriangleSphere::TriangleSphere(const Point3D &_center) : count(0), center(center) {}

/**
 * @brief Generate the sphere.
 * @param center Center of the sphere.
 * @param radius Radius of the sphere.
 * @param latSteps Lateral step of triangles.
 * @param longSteps Longitude step of triangles..
 * @param mat Material of the sphere.
 */
__host__ void TriangleSphere::generate(const Point3D &center, double radius, int latSteps, int longSteps, const Material &mat)
{
    count = 0;
    for (int i = 0; i < latSteps; ++i)
    {
        double theta1 = M_PI * i / latSteps;
        double theta2 = M_PI * (i + 1) / latSteps;

        for (int j = 0; j < longSteps; ++j)
        {
            double phi1 = 2 * M_PI * j / longSteps;
            double phi2 = 2 * M_PI * (j + 1) / longSteps;

            Point3D p1(
                center.x + radius * sin(theta1) * cos(phi1),
                center.y + radius * cos(theta1),
                center.z + radius * sin(theta1) * sin(phi1));

            Point3D p2(
                center.x + radius * sin(theta2) * cos(phi1),
                center.y + radius * cos(theta2),
                center.z + radius * sin(theta2) * sin(phi1));

            Point3D p3(
                center.x + radius * sin(theta2) * cos(phi2),
                center.y + radius * cos(theta2),
                center.z + radius * sin(theta2) * sin(phi2));

            Point3D p4(
                center.x + radius * sin(theta1) * cos(phi2),
                center.y + radius * cos(theta1),
                center.z + radius * sin(theta1) * sin(phi2));

            if (count + 2 <= MAX_SPHERE_TRIANGLES)
            {
                triangles[count++] = Triangle(p1, p2, p3, mat);
                triangles[count++] = Triangle(p1, p3, p4, mat);
            }
        }
    }
}

/**
 * @brief Rotates the sphere around the X axis.
 * @param angle Rotation angle in degrees.
 * @param c Center of rotation.
 */
__host__ __device__ void TriangleSphere::rotateX(double angle, const Point3D &c)
{
    for (int i = 0; i < MAX_SPHERE_TRIANGLES; ++i)
    {
        triangles[i].p1 = rotateAroundX(triangles[i].p1, c, angle);
        triangles[i].p2 = rotateAroundX(triangles[i].p2, c, angle);
        triangles[i].p3 = rotateAroundX(triangles[i].p3, c, angle);
    }
}

/**
 * @brief Rotates the sphere around the Y axis.
 * @param angle Rotation angle in degrees.
 * @param c Center of rotation.
 */
__host__ __device__ void TriangleSphere::rotateY(double angle, const Point3D &c)
{
    for (int i = 0; i < MAX_SPHERE_TRIANGLES; ++i)
    {
        triangles[i].p1 = rotateAroundY(triangles[i].p1, c, angle);
        triangles[i].p2 = rotateAroundY(triangles[i].p2, c, angle);
        triangles[i].p3 = rotateAroundY(triangles[i].p3, c, angle);
    }
}

/**
 * @brief Gets the center of the sphere.
 * @return The center point of the sphere.
 */
__host__ __device__ Point3D TriangleSphere::getCenter() const
{
    return center;
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
