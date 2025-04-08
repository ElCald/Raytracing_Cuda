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
    Material(Vecteur3D _ambient, Vecteur3D _diffuse, Vecteur3D _specular, float _shininess);
};

// Color class
class Color
{
public:
    // Attributes
    int r, g, b;

    // Constructor
    Color(int _r, int _g, int _b);

    // to Int
    int toInt() const;
};

// Forme class
class Form
{
public:
    // Attributes
    Material materiau;

    // Constructor and Destructor
    Form(Material _materiau);
    virtual ~Form() {}

    // Methods
    virtual bool intersection(const Ray &r, double &t) const = 0;
    virtual Vecteur3D getNormal(const Point3D &p) const = 0;
};

// Sphere class
class Sphere : public Form
{
public:
    // Attributes
    Point3D center;
    double rayon;

    // Constructor
    Sphere(Point3D _center, double _rayon, Material _materiau);

    // Methods
    bool intersection(const Ray &r, double &t) const override;
    Vecteur3D getNormal(const Point3D &p) const override;
};

// Triangle class
class Triangle : public Form
{
public:
    // Attributes
    Point3D p1, p2, p3;

    // Constructor
    Triangle(Point3D _p1, Point3D _p2, Point3D _p3, Material _materiau);

    // Methods
    bool intersection(const Ray &r, double &t) const override;
    bool contains(const Point3D &p) const;
    Vecteur3D getNormal(const Point3D &p) const override;
};

// Square class
class Square : public Form
{
public:
    // Attributes
    Point3D p1, p2, p3, p4;

    // Constructor
    Square(Point3D _p1, Point3D _p2, Point3D _p3, Point3D _p4, Material _materiau);

    // Methods
    bool intersection(const Ray &r, double &t) const override;
    Vecteur3D getNormal(const Point3D &p) const override;
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
    Cube(double _size, const Point3D &_center, Material _materiau);

    // Methods
    Point3D getCenter();
    double getSize();
    void rotateX(double angle, Point3D center);
    void rotateY(double angle, Point3D center);
    void rotateZ(double angle, Point3D center);
    void translateX(double direc);
    void translateY(double direc);
    void translateZ(double direc);
    bool intersection(const Ray &r, double &t) const override;
    Vecteur3D getNormal(const Point3D &p) const override;
};

// Some functions to rotate a point
Point3D rotateAroundX(const Point3D &P, const Point3D &O, double angle);
Point3D rotateAroundY(const Point3D &P, const Point3D &O, double angle);
Point3D rotateAroundZ(const Point3D &P, const Point3D &O, double angle);

#endif // FORMS_H