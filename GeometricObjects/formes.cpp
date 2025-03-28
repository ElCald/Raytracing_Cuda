#include "formes.h"
#include <cmath>

using namespace std;


///// ---- SPHERE ---- /////



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


Cube::Cube(double _size, const Point3D &_center, Material _materiau)
    : Forme(_materiau), size(_size), center(_center) {}



vector<Carre> Cube::generateFaces() const {
    vector<Carre> faces;

    // Les 6 faces du cube (en utilisant les points)
    // Face avant
    faces.push_back(Carre(
        Point3D(center.x - size / 2, center.y - size / 2, center.z + size / 2),
        Point3D(center.x + size / 2, center.y - size / 2, center.z + size / 2),
        Point3D(center.x + size / 2, center.y + size / 2, center.z + size / 2),
        Point3D(center.x - size / 2, center.y + size / 2, center.z + size / 2),
        materiau
    ));
    // Face arrière
    faces.push_back(Carre(
        Point3D(center.x - size / 2, center.y - size / 2, center.z - size / 2),
        Point3D(center.x + size / 2, center.y - size / 2, center.z - size / 2),
        Point3D(center.x + size / 2, center.y + size / 2, center.z - size / 2),
        Point3D(center.x - size / 2, center.y + size / 2, center.z - size / 2),
        materiau
    ));
    // Face gauche
    faces.push_back(Carre(
        Point3D(center.x - size / 2, center.y - size / 2, center.z - size / 2),
        Point3D(center.x - size / 2, center.y - size / 2, center.z + size / 2),
        Point3D(center.x - size / 2, center.y + size / 2, center.z + size / 2),
        Point3D(center.x - size / 2, center.y + size / 2, center.z - size / 2),
        materiau
    ));
    // Face droite
    faces.push_back(Carre(
        Point3D(center.x + size / 2, center.y - size / 2, center.z - size / 2),
        Point3D(center.x + size / 2, center.y - size / 2, center.z + size / 2),
        Point3D(center.x + size / 2, center.y + size / 2, center.z + size / 2),
        Point3D(center.x + size / 2, center.y + size / 2, center.z - size / 2),
        materiau
    ));
    // Face supérieure
    faces.push_back(Carre(
        Point3D(center.x - size / 2, center.y + size / 2, center.z - size / 2),
        Point3D(center.x + size / 2, center.y + size / 2, center.z - size / 2),
        Point3D(center.x + size / 2, center.y + size / 2, center.z + size / 2),
        Point3D(center.x - size / 2, center.y + size / 2, center.z + size / 2),
        materiau
    ));
    // Face inférieure
    faces.push_back(Carre(
        Point3D(center.x - size / 2, center.y - size / 2, center.z - size / 2),
        Point3D(center.x + size / 2, center.y - size / 2, center.z - size / 2),
        Point3D(center.x + size / 2, center.y - size / 2, center.z + size / 2),
        Point3D(center.x - size / 2, center.y - size / 2, center.z + size / 2),
        materiau
    ));

    return faces;
}



bool Cube::intersection(const Ray &r, double &t) const {
    // Teste l'intersection avec les 6 faces du cube
    std::vector<Carre> faces = generateFaces();
    double minT = std::numeric_limits<double>::infinity();
    bool hit = false;

    for (const Carre &face : faces) {
        double t_face;
        if (face.intersection(r, t_face) && t_face < minT) {
            minT = t_face;
            hit = true;
        }
    }

    if (hit) {
        t = minT;
        return true;
    }

    return false;
}


Vecteur3D Cube::getNormal(const Point3D &p) const {
    // Calcul de la normale d'une face du cube
    // Il suffit de tester sur quelle face l'intersection a eu lieu
    std::vector<Carre> faces = generateFaces();
    double t;
    for (const Carre &face : faces) {
        if (face.intersection(Ray(p, Vecteur3D(0, 0, 0)), t)) { // Test intersection avec la face
            return face.getNormal(p);
        }
    }
    return Vecteur3D(0, 0, 0); // Valeur par défaut, ne devrait pas arriver
}



// Appliquer une rotation autour de l'axe X
void Cube::rotateX(double angle) {
    double rad = angle * M_PI / 180.0;  // Convertir l'angle en radians
    std::vector<Point3D> corners = getCorners();  // Obtenir les coins du cube

    for (Point3D &p : corners) {
        double y = p.y;
        double z = p.z;
        p.y = y * std::cos(rad) - z * std::sin(rad);
        p.z = y * std::sin(rad) + z * std::cos(rad);
    }
}

// Appliquer une rotation autour de l'axe Y
void Cube::rotateY(double angle) {
    double rad = angle * M_PI / 180.0;  // Convertir en radians
    std::vector<Point3D> corners = getCorners();  // Obtenir les coins du cube

    // Rotation autour de l'axe Y
    for (Point3D &p : corners) {
        double x = p.x;
        double z = p.z;
        p.x = x * std::cos(rad) + z * std::sin(rad);
        p.z = -x * std::sin(rad) + z * std::cos(rad);
    }
}

// Appliquer une rotation autour de l'axe Z
void Cube::rotateZ(double angle) {
    double rad = angle * M_PI / 180.0;  // Convertir en radians
    std::vector<Point3D> corners = getCorners();  // Obtenir les coins du cube

    // Rotation autour de l'axe Z
    for (Point3D &p : corners) {
        double x = p.x;
        double y = p.y;
        p.x = x * std::cos(rad) - y * std::sin(rad);
        p.y = x * std::sin(rad) + y * std::cos(rad);
    }
}


// Fonction pour obtenir les coins du cube
std::vector<Point3D> Cube::getCorners() const {
    std::vector<Point3D> corners;
    double halfSize = size / 2.0;
    
    // Crée les 8 coins du cube (cube centré autour de center)
    corners.push_back(Point3D(center.x - halfSize, center.y - halfSize, center.z - halfSize));
    corners.push_back(Point3D(center.x + halfSize, center.y - halfSize, center.z - halfSize));
    corners.push_back(Point3D(center.x + halfSize, center.y + halfSize, center.z - halfSize));
    corners.push_back(Point3D(center.x - halfSize, center.y + halfSize, center.z - halfSize));
    corners.push_back(Point3D(center.x - halfSize, center.y - halfSize, center.z + halfSize));
    corners.push_back(Point3D(center.x + halfSize, center.y - halfSize, center.z + halfSize));
    corners.push_back(Point3D(center.x + halfSize, center.y + halfSize, center.z + halfSize));
    corners.push_back(Point3D(center.x - halfSize, center.y + halfSize, center.z + halfSize));

    return corners;
}