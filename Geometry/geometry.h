#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cmath>

// Classe représentant un vecteur 3D
class Vecteur3D
{
public:
    double x, y, z;

    Vecteur3D(double _x = 0, double _y = 0, double _z = 0);

    double dot(const Vecteur3D &v) const;      // Produit scalaire
    Vecteur3D cross(const Vecteur3D &v) const; // Produit vectoriel
    double length() const;                     // Norme du vecteur
    Vecteur3D normalized() const;              // Vecteur unitaire

    // Opérateurs pour les calculs vectoriels
    Vecteur3D operator+(const Vecteur3D &v) const;
    Vecteur3D operator-(const Vecteur3D &v) const;
    Vecteur3D operator*(double scalar) const;
    Vecteur3D operator/(double scalar) const;
    Vecteur3D operator*(const Vecteur3D &v) const;
};

// Classe représentant un point dans l'espace 3D
class Point3D
{
public:
    double x, y, z;

    Point3D(double _x = 0, double _y = 0, double _z = 0);

    Vecteur3D operator-(const Point3D &p) const;
    Point3D operator+(const Point3D &p) const;
    Point3D operator+(const Vecteur3D &v) const;
};

// Classe représentant un rayon (Raytracing)
class Ray
{
public:
    Point3D origine;
    Vecteur3D direction;

    Ray(const Point3D &orig, const Vecteur3D &direc);
    Point3D at(double t) const; // Renvoie un point sur le rayon à une distance t
};

#endif // GEOMETRY_H
