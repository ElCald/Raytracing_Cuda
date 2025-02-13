#ifndef GEOMETRY_H
#define GEOMETRY_H

class Point3D {

    public:
        Point3D();
        Point3D(double _x, double _y, double _z);
        virtual ~Point3D() = default;


        double x;
        double y;
        double z;

};


class Vecteur3D {

    public:
        Vecteur3D();
        Vecteur3D(double _x, double _y, double _z);
        virtual ~Vecteur3D() = default;

        double x;
        double y;
        double z;


        double dot(const Vecteur3D& v) const; // Produit scalaire
        Vecteur3D cross(const Vecteur3D& v) const; //

        double length() const; // Longueur du vecteur
        Vecteur3D normalized() const; // Vecteur normalisé

        // Opérations sur vecteurs
        Vecteur3D operator+(const Vecteur3D& v) const;
        Vecteur3D operator-(const Vecteur3D& v) const;
        Vecteur3D operator*(double scalar) const;
        Vecteur3D operator/(double scalar) const;
        Vecteur3D operator-() const;

};


class Ray {

    public:
        Ray();
        Ray(const Point3D& orig, const Vecteur3D& direc);
        ~Ray() = default;

        Point3D at(double t) const;  // Renvoie un point sur le rayon à une distance t

        Point3D origine;
        Vecteur3D direction;
};


#endif