#include "camera.h"
#include <cmath>

Camera::Camera(Point3D _position, Vecteur3D _direction, double _fov, int _width, int _height)
    : position(_position), direction(_direction.normalized()), fov(_fov), width(_width), height(_height)
{

    right = Vecteur3D(1, 0, 0);
    up = Vecteur3D(0, 1, 0);
}

Ray Camera::generateRay(int x, int y) const
{
    double aspect_ratio = (double)width / height;
    double scale = tan(fov * 0.5 * M_PI / 180);

    double pixelNDC_x = (x + 0.5) / width;
    double pixelNDC_y = (y + 0.5) / height;

    double px = (2 * pixelNDC_x - 1) * aspect_ratio * scale;
    double py = (1 - 2 * pixelNDC_y) * scale;

    Vecteur3D ray_direction = (direction + right * px + up * py).normalized();
    return Ray(position, ray_direction);
}




void Camera::rotateX(double angle)
{
    direction.x += angle;
}

void Camera::rotateY(double angle)
{
    direction.y += angle;
}



void Camera::rotatePosX(double angle, Point3D centre)
{
    position = rotateAroundX(position, centre, angle);
}

void Camera::rotatePosY(double angle, Point3D centre)
{
    position = rotateAroundY(position, centre, angle);
}

void Camera::rotatePosZ(double angle, Point3D centre)
{
    position = rotateAroundZ(position, centre, angle);
}



void Camera::translateX(double direc)
{
    position.x += direc;
}

void Camera::translateY(double direc)
{
    position.y += direc;
}

void Camera::translateZ(double direc)
{
    position.z += direc;
}



