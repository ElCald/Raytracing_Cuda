#ifndef FORMES_H
#define FORMES_H

#include "../Geometry/geometry.h"


class Forme {
    
    public:
        Forme();
        virtual ~Forme() = default;

        virtual bool intersection(const Ray& ray, double& t) const = 0;

        int materiau;
        int texture;    
};



class Sphere : public Forme {

    public:
        Sphere();
        Sphere(Point3D _centre, double _rayon);
        virtual ~Sphere() = default;

        bool intersection(const Ray& ray, double& t) const override;  // Vérifie si un rayon touche la sphère
        Vecteur3D getNormal(const Point3D& p) const;  // Renvoie la normale en un point donné de la sphère

        double rayon;
        Point3D centre;
};


class Triangle : public Forme {

    public:
        Triangle();
        virtual ~Triangle() = default;

        bool intersection(const Ray& ray, double& t) const override;
};

#endif