// geometry.h
#ifndef GEOMETRY_H
#define GEOMETRY_H

// Includes
#include <math.h>

// 3D Vector struct
struct Vecteur3D
{
  // Attributes
  double x, y, z;

  // Constructor
  __host__ __device__ Vecteur3D(double _x = 0, double _y = 0, double _z = 0);

  // Methods
  __host__ __device__ double dot(const Vecteur3D &v) const;
  __host__ __device__ double length() const;
  __host__ __device__ Vecteur3D cross(const Vecteur3D &v) const;
  __host__ __device__ Vecteur3D normalized() const;
  __host__ __device__ Vecteur3D inverse() const;

  // Operators
  __host__ __device__ Vecteur3D operator+(const Vecteur3D &v) const;
  __host__ __device__ Vecteur3D operator-(const Vecteur3D &v) const;
  __host__ __device__ Vecteur3D operator*(double scalar) const;
  __host__ __device__ Vecteur3D operator/(double scalar) const;
  __host__ __device__ Vecteur3D operator*(const Vecteur3D &v) const;
  __host__ __device__ Vecteur3D &operator+=(const Vecteur3D &v);
};

// 3D Point struct
struct Point3D
{
  // Attributes
  double x, y, z;

  // Constructor
  __host__ __device__ Point3D(double _x = 0, double _y = 0, double _z = 0);

  // Operators
  __host__ __device__ Vecteur3D operator-(const Point3D &p) const;
  __host__ __device__ Point3D operator+(const Point3D &p) const;
  __host__ __device__ Point3D operator+(const Vecteur3D &v) const;
};

// Ray struct
struct Ray
{
  // Attributes
  Point3D origine;
  Vecteur3D direction;

  // Constructor
  __host__ __device__ Ray(const Point3D &orig, const Vecteur3D &direc);

  // Method
  __host__ __device__ Point3D at(double t) const;
};

#endif // GEOMETRY_H