#ifndef FORMES_H
#define FORMES_H

#include "../Geometry/geometry.h"


class Forme {
    
    public:
        Forme();
        virtual ~Forme() = default;

        virtual bool intersection(const Point3D p) const = 0;

        int materiau;
    
};



class Sphere : public Forme {

    public:
        Sphere();
        Sphere(Point3D _centre, double _rayon, int _materiau);
        virtual ~Sphere() = default;

        bool intersection(const Point3D p) const override;

        double rayon;
        Point3D centre;
};


class Triangle : public Forme {

    public:
        Triangle();
        virtual ~Triangle() = default;

        bool intersection(const Point3D p) const override;
};

#endif