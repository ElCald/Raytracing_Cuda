// forms.h
#ifndef FORMS_H
#define FORMS_H

// Includes
#include "../Geometry/geometry.h"
#include <vector>

using namespace std;

// Material Class
class Material
{
public:
    // Attributes
    Vecteur3D ambient, diffuse, specular;
    float shininess;

    // Constructor
    __host__ __device__ Material(Vecteur3D _ambient, Vecteur3D _diffuse, Vecteur3D _specular, float _shininess);
};

// Color class
class Color
{
public:
    // Attributes
    int r, g, b;

    // Constructor
    __host__ __device__ Color();
    __host__ __device__ Color(int _r, int _g, int _b);

    // to Int
    __host__ __device__ int toInt() const;
};

// Forme class
class Form
{
public:
    // Attributes
    Material materiau;

    // Constructor and Destructor
    __host__ __device__ Form(Material _materiau);
    __host__ __device__ virtual ~Form() {}

    // Methods
    __host__ __device__ virtual bool intersection(const Ray &r, double &t) const = 0;
    __host__ __device__ virtual Vecteur3D getNormal(const Point3D &p) const = 0;
};

// Sphere class
class Sphere : public Form
{
public:
    // Attributes
    Point3D center;
    double rayon;

    // Constructor
    __host__ __device__ Sphere(Point3D _center, double _rayon, Material _materiau);

    // Methods
    __host__ __device__ bool intersection(const Ray &r, double &t) const override;
    __host__ __device__ Vecteur3D getNormal(const Point3D &p) const override;
};

// Triangle class
class Triangle : public Form
{
public:
    // Attributes
    Point3D p1, p2, p3;

    // Constructor
    __host__ __device__ Triangle(Point3D _p1, Point3D _p2, Point3D _p3, Material _materiau);

    // Methods
    __host__ __device__ bool intersection(const Ray &r, double &t) const override;
    __host__ __device__ bool contains(const Point3D &p) const;
    __host__ __device__ Vecteur3D getNormal(const Point3D &p) const override;
};

// Square class
class Square : public Form
{
public:
    // Attributes
    Point3D p1, p2, p3, p4;

    // Constructor
    __host__ __device__ Square(Point3D _p1, Point3D _p2, Point3D _p3, Point3D _p4, Material _materiau);

    // Methods
    __host__ __device__ bool intersection(const Ray &r, double &t) const override;
    __host__ __device__ Vecteur3D getNormal(const Point3D &p) const override;
};

// Cube class
class Cube : public Form
{
private:
    // Attributes
    double size;
    Point3D center;

public:
    // Attributes
    vector<Triangle *> cube;

    // Constructor
    __host__ __device__ Cube(double _size, const Point3D &_center, Material _materiau);

    // Methods
    __host__ __device__ Point3D getCenter();
    __host__ __device__ double getSize();
    __host__ __device__ void rotateX(double angle, Point3D center);
    __host__ __device__ void rotateY(double angle, Point3D center);
    __host__ __device__ void rotateZ(double angle, Point3D center);
    __host__ __device__ void translateX(double direc);
    __host__ __device__ void translateY(double direc);
    __host__ __device__ void translateZ(double direc);
    __host__ __device__ bool intersection(const Ray &r, double &t) const override;
    __host__ __device__ Vecteur3D getNormal(const Point3D &p) const override;
};

// Some functions to rotate a point
__host__ __device__ Point3D rotateAroundX(const Point3D &P, const Point3D &O, double angle);
__host__ __device__ Point3D rotateAroundY(const Point3D &P, const Point3D &O, double angle);
__host__ __device__ Point3D rotateAroundZ(const Point3D &P, const Point3D &O, double angle);

#endif // FORMS_H