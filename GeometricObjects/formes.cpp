#include "formes.h"
#include <cmath>

using namespace std;


///// ---- SPHERE ---- /////


Sphere::Sphere(Point3D _centre, double _rayon, Material _materiau): Forme(_materiau), centre(_centre), rayon(_rayon) {}


// Intersection d'un rayon avec une sphère
bool Sphere::intersection(const Ray &r, double &t) const
{
    // Le rayon est donné par : r(t) = origine + t * direction
    // On substitue r(t) dans l'équation de la sphère

    // Calcul de l'élément de produit scalaire
    Vecteur3D oc = r.origine - centre;
    double a = r.direction.dot(r.direction);
    double b = 2.0 * oc.dot(r.direction);
    double c = oc.dot(oc) - rayon * rayon;

    // Calcul du discriminant
    double discriminant = b * b - 4 * a * c;

    if (discriminant > 0)
    {
        // Il y a deux intersections possibles
        double t0 = (-b - sqrt(discriminant)) / (2.0 * a);
        double t1 = (-b + sqrt(discriminant)) / (2.0 * a);

        // On choisit le plus proche point d'intersection (t > 0)
        if (t0 > 0 && t1 > 0)
        {
            t = (t0 < t1) ? t0 : t1;
            return true;
        }
        else if (t0 > 0)
        {
            t = t0;
            return true;
        }
        else if (t1 > 0)
        {
            t = t1;
            return true;
        }
    }
    return false;
}


Vecteur3D Sphere::getNormal(const Point3D& p) const {
    return (p - this->centre).normalized();
}




///// ---- TRIANGLE ---- /////

Triangle::Triangle(Point3D _p1, Point3D _p2, Point3D _p3, Material _materiau) : Forme(_materiau), p1(_p1), p2(_p2), p3(_p3) {}


// Intersection d'un rayon avec un triangle
bool Triangle::intersection(const Ray &r, double &t) const
{
    // Vecteurs pour les bords du triangle
    Vecteur3D e1 = p2 - p1;
    Vecteur3D e2 = p3 - p1;

    // Calcul du produit vectoriel entre la direction du rayon et l'un des bords
    Vecteur3D h = r.direction.cross(e2);
    double a = e1.dot(h);

    // Si le produit scalaire est proche de zéro, il n'y a pas d'intersection
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

    // Calcul de t, la distance de l'intersection
    t = f * e2.dot(q);

    if (t > 1e-6)
        return true;
    else
        return false;
}


Vecteur3D Triangle::getNormal(const Point3D& p) const {
    // Les vecteurs AB et AC
    Vecteur3D AB = p2 - p1;
    Vecteur3D AC = p3 - p1;
    // Le produit vectoriel entre AB et AC donne la normale
    return AB.cross(AC).normalized();
}

// Vérifie si un point est à l'intérieur du triangle
bool Triangle::contains(const Point3D& p) const {
    // Vecteurs formés par les côtés du triangle
    Vecteur3D v0 = p2 - p1;
    Vecteur3D v1 = p3 - p1;
    Vecteur3D v2 = p - p1;

    // Calcul des produits scalaires
    double dot00 = v0.dot(v0);
    double dot01 = v0.dot(v1);
    double dot02 = v0.dot(v2);
    double dot11 = v1.dot(v1);
    double dot12 = v1.dot(v2);

    // Calcul du dénominateur (invariant)
    double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);

    // Calcul des barycentriques
    double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    // Le point est à l'intérieur si u >= 0, v >= 0, et u + v <= 1
    return (u >= 0) && (v >= 0) && (u + v <= 1);
}


///// ---- CARRE ---- /////


// Intersection d'un rayon avec un carré (simplifié comme 4 bords)
bool Carre::intersection(const Ray &r, double &t) const
{
    // Vous devez vérifier l'intersection avec chaque côté du carré.
    // Par exemple, en décomposant le carré en 2 triangles et en vérifiant
    // si le rayon intersecte l'un ou l'autre.

    // Exemple avec une méthode d'intersection avec des triangles
    // Divisez le carré en 2 triangles et vérifiez chaque triangle
    Triangle triangle1(p1, p2, p3, {Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0, 0, 0), Vecteur3D(1, 1, 1), 32}); // Premier triangle du carré
    Triangle triangle2(p1, p3, p4, {Vecteur3D(0.1, 0.1, 0.1), Vecteur3D(0, 0, 0), Vecteur3D(1, 1, 1), 32}); // Deuxième triangle du carré

    double t1, t2;
    if (triangle1.intersection(r, t1) || triangle2.intersection(r, t2))
    {
        t = std::min(t1, t2); // Choisir le plus proche
        return true;
    }
    return false;
}


Vecteur3D Carre::getNormal(const Point3D& p) const {
    // On prend les trois premiers points pour définir le plan
    Vecteur3D AB = p2 - p1;
    Vecteur3D AD = p4 - p1;
    // Le produit vectoriel entre AB et AD donne la normale
    return AB.cross(AD).normalized();
}



///// ---- CUBE ---- /////
Cube::Cube(double _size, const Point3D &_center, Material _materiau) : size(_size), center(_center), Forme(_materiau) {

    // face devant
    Triangle *t1 = new Triangle(Point3D(0, 0, 0), Point3D(_size, 0, 0), Point3D(_size, _size, 0), _materiau);
    cube.push_back(t1);

    Triangle *t2 = new Triangle(Point3D(_size, _size, 0), Point3D(0, _size, 0), Point3D(0, 0, 0), _materiau);
    cube.push_back(t2);

    // face haut
    Triangle *t3 = new Triangle(Point3D(0, _size, 0), Point3D(_size, _size, 0), Point3D(_size, _size, -_size), _materiau);
    cube.push_back(t3);

    Triangle *t4 = new Triangle(Point3D(_size, _size, -_size), Point3D(0, _size, -_size), Point3D(0, _size, 0), _materiau);
    cube.push_back(t4);

    // //face droite
    Triangle *t5 = new Triangle(Point3D(_size, 0, 0), Point3D(_size, 0, -_size), Point3D(_size, _size, -_size), _materiau);
    cube.push_back(t5);

    Triangle *t6 = new Triangle(Point3D(_size, _size, -_size), Point3D(_size, _size, 0), Point3D(_size, 0, 0), _materiau);
    cube.push_back(t6);

    // face gauche
    Triangle *t7 = new Triangle(Point3D(0, 0, -_size), Point3D(0, 0, 0), Point3D(0, _size, 0), _materiau);
    cube.push_back(t7);

    Triangle *t8 = new Triangle(Point3D(0, _size, 0), Point3D(0, _size, -_size), Point3D(0, 0, -_size), _materiau);
    cube.push_back(t8);

    // face bas
    Triangle *t9 = new Triangle(Point3D(0, 0, -_size), Point3D(_size, 0, -_size), Point3D(_size, 0, 0), _materiau);
    cube.push_back(t9);

    Triangle *t10 = new Triangle(Point3D(_size, 0, 0), Point3D(0, 0, 0), Point3D(0, 0, -_size), _materiau);
    cube.push_back(t10);

    // face arriere
    Triangle *t11 = new Triangle(Point3D(0, _size, -_size), Point3D(_size, _size, -_size), Point3D(_size, 0, -_size), _materiau);
    cube.push_back(t11);

    Triangle *t12 = new Triangle(Point3D(_size, 0, -_size), Point3D(0, 0, -_size), Point3D(0, _size, -_size), _materiau);
    cube.push_back(t12);

    int x_temp=center.x, y_temp=center.y, z_temp=center.z;

    translateX(center.x);
    translateY(center.y);
    translateZ(center.z);

    center.x = x_temp;
    center.y = y_temp;
    center.z = z_temp;
}



void Cube::translateX(double direc){
    for(auto t : cube){
        t->p1.x += direc;
        t->p2.x += direc;
        t->p3.x += direc;
    }

    center.x += direc;
}


void Cube::translateY(double direc){
    for(auto t : cube){
        t->p1.y += direc;
        t->p2.y += direc;
        t->p3.y += direc;
    }

    center.y += direc;
}


void Cube::translateZ(double direc){
    for(auto t : cube){
        t->p1.z += direc;
        t->p2.z += direc;
        t->p3.z += direc;
    }

    center.z += direc;
}


void Cube::rotateX(double angle){
    for(auto t : cube){
        t->p1 = rotateAroundX(t->p1, center,  angle);
        t->p2 = rotateAroundX(t->p2, center,  angle);
        t->p3 = rotateAroundX(t->p3, center,  angle);
    }
}


void Cube::rotateY(double angle){
    for(auto t : cube){
        t->p1 = rotateAroundY(t->p1, center,  angle);
        t->p2 = rotateAroundY(t->p2, center,  angle);
        t->p3 = rotateAroundY(t->p3, center,  angle);
    }
}


void Cube::rotateZ(double angle){
    for(auto t : cube){
        t->p1 = rotateAroundZ(t->p1, center,  angle);
        t->p2 = rotateAroundZ(t->p2, center,  angle);
        t->p3 = rotateAroundZ(t->p3, center,  angle);
    }
}



// implémentation par défaut
bool Cube::intersection(const Ray &r, double &t) const {
    return false;
}
Vecteur3D Cube::getNormal(const Point3D& p) const {
    return Vecteur3D();
}





///// ---- AUTRE ---- /////

Point3D rotateAroundX(const Point3D& P, const Point3D& O, double angle) {
    // Convertir l'angle en radians
    double rad = angle * M_PI / 180.0;
    double cosAngle = cos(rad);
    double sinAngle = sin(rad);

    // Traduire le point P pour que O soit à l'origine
    double dx = P.x - O.x;
    double dy = P.y - O.y;
    double dz = P.z - O.z;

    // Appliquer la matrice de rotation autour de l'axe X
    double newY = cosAngle * dy - sinAngle * dz;
    double newZ = sinAngle * dy + cosAngle * dz;

    // Revenir à la position d'origine
    return Point3D(P.x, newY + O.y, newZ + O.z);
}


Point3D rotateAroundY(const Point3D& P, const Point3D& O, double angle) {
    // Convertir l'angle en radians
    double rad = angle * M_PI / 180.0;
    double cosAngle = cos(rad);
    double sinAngle = sin(rad);

    // Traduire le point P pour que O soit à l'origine
    double dx = P.x - O.x;
    double dy = P.y - O.y;
    double dz = P.z - O.z;

    // Appliquer la matrice de rotation autour de l'axe Y
    double newX = cosAngle * dx + sinAngle * dz;
    double newZ = -sinAngle * dx + cosAngle * dz;

    // Revenir à la position d'origine
    return Point3D(newX + O.x, P.y, newZ + O.z);
}


Point3D rotateAroundZ(const Point3D& P, const Point3D& O, double angle) {
    // Convertir l'angle en radians
    double rad = angle * M_PI / 180.0;
    double cosAngle = cos(rad);
    double sinAngle = sin(rad);

    // Traduire le point P pour que O soit à l'origine
    double dx = P.x - O.x;
    double dy = P.y - O.y;

    // Appliquer la matrice de rotation autour de l'axe Z
    double newX = cosAngle * dx - sinAngle * dy;
    double newY = sinAngle * dx + cosAngle * dy;

    // Revenir à la position d'origine
    return Point3D(newX + O.x, newY + O.y, P.z);
}
