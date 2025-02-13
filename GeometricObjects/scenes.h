#ifndef SCENES_H
#define SCENES_H

#include <vector>

#include "formes.h"
#include "../Cameras/camera.h"

using namespace std;

class Scenes {
    public:
        Scenes();
        ~Scenes() = default;

        Camera& getCamera();

    private:
        vector<Forme> objets;
        Camera camera;

};

#endif