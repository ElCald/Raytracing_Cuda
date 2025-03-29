// formes.h
#ifndef FORMES_H
#define FORMES_H

#include "../Geometry/geometry.h"
#include <vector>

using namespace std;

class Material {
public:
    Material(Vecteur3D _ambient, Vecteur3D _diffuse, Vecteur3D _specular, float _shininess) : ambient(_ambient), diffuse(_diffuse), specular(_specular), shininess(_shininess) {}
    Vecteur3D ambient;
    Vecteur3D diffuse; // couleur
    Vecteur3D specular;
    float shininess;
};

class Color
{
public:
    int r, g, b;

    Color(int r = 0, int g = 0, int b = 0) : r(r), g(g), b(b) {}

    // Convertir la couleur en un entier représentant les 3 canaux RVB
    int toInt() const
    {
        return (r << 16) | (g << 8) | b;
    }
};

class Forme
{
public:
    Material materiau; // Ajouter une couleur à chaque forme

    Forme(Material _materiau) : materiau(_materiau) {}

    virtual bool intersection(const Ray &r, double &t) const = 0;
    virtual Vecteur3D getNormal(const Point3D& p) const = 0;
};

class Sphere : public Forme
{
public:
    Point3D centre;
    double rayon;

    Sphere(Point3D _centre, double _rayon, Material _materiau);

    bool intersection(const Ray &r, double &t) const override;
    Vecteur3D getNormal(const Point3D& p) const override;
};

class Triangle : public Forme
{
public:
    Point3D p1, p2, p3;

    Triangle(Point3D _p1, Point3D _p2, Point3D _p3, Material _materiau);

    bool intersection(const Ray &r, double &t) const override;
    Vecteur3D getNormal(const Point3D& p) const override;
    bool contains(const Point3D& p) const;
};


class Carre : public Forme
{
public:
    Point3D p1, p2, p3, p4;

    Carre(Point3D _p1, Point3D _p2, Point3D _p3, Point3D _p4, Material _materiau);

    bool intersection(const Ray &r, double &t) const override;
    Vecteur3D getNormal(const Point3D& p) const override;
};




class Cube : public Forme {
private:
    double size;   // Taille du cube
    Point3D center; // Origine > point en bas à gauche de la face avant (0,0,0)

public:
    vector<Triangle*> cube;

    Cube(double _size, const Point3D &_center, Material _materiau);


    // Rotation autour de l'axe X
    void rotateX(double angle);
    // Rotation autour de l'axe Y
    void rotateY(double angle);
    // Rotation autour de l'axe Z
    void rotateZ(double angle);

    // Translation en X
    void translateX(double direc);
    // Translation en Y
    void translateY(double direc);
    // Translation en Z
    void translateZ(double direc);

    bool intersection(const Ray &r, double &t) const override;
    Vecteur3D getNormal(const Point3D& p) const override;
};

Point3D rotateAroundX(const Point3D& P, const Point3D& O, double angle);
Point3D rotateAroundY(const Point3D& P, const Point3D& O, double angle);
Point3D rotateAroundZ(const Point3D& P, const Point3D& O, double angle);




#endif // FORMES_H