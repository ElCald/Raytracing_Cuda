// forms.h
#ifndef FORMS_H
#define FORMS_H

// Includes
#include "../Geometry/geometry.h"
#include <vector>

#define MAX_SPHERE_TRIANGLES 1024
#define MAX_PYRAMIDE_TRIANGLES 4

using namespace std;

// Material struct
struct Material
{
  // Attributes
  Vecteur3D ambient, diffuse, specular;
  float shininess;

  // Constructors
  __host__ __device__ Material() {}
  __host__ __device__ Material(Vecteur3D _a, Vecteur3D _d, Vecteur3D _s,
                               float _sh);
};

// Color struct
struct Color
{
  // Attributes
  int r, g, b;

  // Constructors
  __host__ __device__ Color();

  // toInt
  __host__ __device__ Color(int _r, int _g, int _b);
  __host__ __device__ int toInt() const;
};

// Triangle struct
struct Triangle
{
  // Attributes
  Point3D p1, p2, p3;
  Material mat;

  // Constructors
  __host__ __device__ Triangle() {}
  __host__ __device__ Triangle(Point3D _p1, Point3D _p2, Point3D _p3,
                               Material _mat);

  // Methods
  __host__ __device__ bool intersection(const Ray &r, double &t) const;
  __host__ __device__ bool contains(const Point3D &p) const;
  __host__ __device__ Vecteur3D getNormal(const Point3D &p) const;
};

// Cube struct
struct Cube
{
  // Attributes
  Triangle triangles[12];
  double size;
  Point3D center;

  // Constructors
  __host__ __device__ Cube() {}
  __host__ __device__ Cube(double _size, const Point3D &_center,
                           const Material &_mat);

  // Methods
  __host__ __device__ void rotateX(double angle, const Point3D &center);
  __host__ __device__ void rotateY(double angle, const Point3D &center);
  __host__ __device__ void rotateZ(double angle, const Point3D &center);
  __host__ __device__ void translateX(double val);
  __host__ __device__ void translateY(double val);
  __host__ __device__ void translateZ(double val);

  __host__ __device__ Point3D getCenter() const;
  __host__ __device__ double getSize() const;
};

// Pyramid struct
struct Pyramid
{
  // Attributes
  Triangle triangles[4];

  // Constructor
  __host__ __device__ Pyramid(const Point3D &base1, const Point3D &base2,
                              const Point3D &base3, const Point3D &apex,
                              const Material &mat);
};

// Sphere struct
struct TriangleSphere
{
  // Attributes
  Triangle triangles[MAX_SPHERE_TRIANGLES];
  int count;

  // Constructors
  __host__ TriangleSphere();

  // Methods
  __host__ void generate(const Point3D &center, double radius, int latSteps,
                         int longSteps, const Material &mat);
};

// Helpers
__host__ __device__ Point3D rotateAroundX(const Point3D &P, const Point3D &O,
                                          double angle);
__host__ __device__ Point3D rotateAroundY(const Point3D &P, const Point3D &O,
                                          double angle);
__host__ __device__ Point3D rotateAroundZ(const Point3D &P, const Point3D &O,
                                          double angle);

#endif // FORMS_H