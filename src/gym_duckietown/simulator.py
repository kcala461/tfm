import itertools
import os
from collections import namedtuple
from ctypes import POINTER
from dataclasses import dataclass

import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from typing import Any, cast, Dict, List, NewType, Optional, Sequence, Tuple, Union

import geometry
import geometry as g
import gym
import math
import numpy as np
import pyglet
import yaml
from geometry import SE2value
from gym import spaces
from gym.utils import seeding
from numpy.random.mtrand import RandomState
from pyglet import gl, image, window

from duckietown_world import (
    get_DB18_nominal,
    get_DB18_uncalibrated,
    get_texture_file,
    MapFormat1,
    MapFormat1Constants,
    MapFormat1Constants as MF1C,
    MapFormat1Object,
    SE2Transform,
)
from duckietown_world.gltf.export import get_duckiebot_color_from_colorname
from duckietown_world.resources import get_resource_path
from duckietown_world.world_duckietown.map_loading import get_transform
from . import logger
from .check_hw import get_graphics_information
from .collision import (
    agent_boundbox,
    generate_norm,
    intersects,
    safety_circle_intersection,
    safety_circle_overlap,
    tile_corners,
)
from .distortion import Distortion
from .exceptions import InvalidMapException, NotInLane
from .graphics import (
    bezier_closest,
    bezier_draw,
    bezier_point,
    bezier_tangent,
    create_frame_buffers,
    gen_rot_matrix,
    load_texture,
    Texture,
)
from .objects import CheckerboardObj, DuckiebotObj, DuckieObj, TrafficLightObj, WorldObj
from .objmesh import get_mesh, MatInfo, ObjMesh
from .randomization import Randomizer
from .utils import get_subdir_path

DIM = 0.5

TileKind = NewType("TileKind", str)


class TileDict(TypedDict):
    # {"coords": (i, j), "kind": kind, "angle": angle, "drivable": drivable})
    coords: Tuple[int, int]
    kind: TileKind
    angle: int
    drivable: bool
    texture: Texture
    color: np.ndarray
    curves: Any


@dataclass
class DoneRewardInfo:
    done: bool
    done_why: str
    done_code: str
    reward: float


@dataclass
class DynamicsInfo:
    motor_left: float
    motor_right: float


# Rendering window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Camera image size
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480

# Blue sky horizon color
BLUE_SKY = np.array([0.45, 0.82, 1])

# Color meant to approximate interior walls
WALL_COLOR = np.array([0.64, 0.71, 0.28])

# np.array([0.15, 0.15, 0.15])
GREEN = (0.0, 1.0, 0.0)
# Ground/floor color


# Angle at which the camera is pitched downwards
CAMERA_ANGLE = 19.15

# Camera field of view angle in the Y direction
# Note: robot uses Raspberri Pi camera module V1.3
# https://www.raspberrypi.org/documentation/hardware/camera/README.md
CAMERA_FOV_Y = 75

# Distance from camera to floor (10.8cm)
CAMERA_FLOOR_DIST = 0.108

# Forward distance between the camera (at the front)
# and the center of rotation (6.6cm)
CAMERA_FORWARD_DIST = 0.066

# Distance (diameter) between the center of the robot wheels (10.2cm)
WHEEL_DIST = 0.102

# Total robot width at wheel base, used for collision detection
# Note: the actual robot width is 13cm, but we add a litte bit of buffer
#       to faciliate sim-to-real transfer.
ROBOT_WIDTH = 0.13 + 0.02

# Total robot length
# Note: the center of rotation (between the wheels) is not at the
#       geometric center see CAMERA_FORWARD_DIST
ROBOT_LENGTH = 0.18

# Height of the robot, used for scaling
ROBOT_HEIGHT = 0.12

# Safety radius multiplier
SAFETY_RAD_MULT = 4

# Robot safety circle radius
AGENT_SAFETY_RAD = (max(ROBOT_LENGTH, ROBOT_WIDTH) / 2) * SAFETY_RAD_MULT

# Minimum distance spawn position needs to be from all objects
MIN_SPAWN_OBJ_DIST = 0.25

# Road tile dimensions (2ft x 2ft, 61cm wide)
# self.road_tile_size = 0.61

# Maximum forward robot speed in meters/second
DEFAULT_ROBOT_SPEED = 1.20 
# approx 2 tiles/second

DEFAULT_FRAMERATE = 30

DEFAULT_MAX_STEPS = 1500

DEFAULT_MAP_NAME = "udem1"

DEFAULT_FRAME_SKIP = 1

DEFAULT_ACCEPT_START_ANGLE_DEG = 60

REWARD_INVALID_POSE = -500
REWARD_SUCCESS=600

MAX_SPAWN_ATTEMPTS = 5000

LanePosition0 = namedtuple("LanePosition", "dist dot_dir angle_deg angle_rad")


# Convierte a Json los datos 4 datos de la tupla dist, dot_dir, angle_deg, angle_rad
class LanePosition(LanePosition0):
    def as_json_dict(self):
        """Serialization-friendly format."""
        return dict(dist=self.dist, dot_dir=self.dot_dir, angle_deg=self.angle_deg, angle_rad=self.angle_rad)


# Crea la clase simulador
class Simulator(gym.Env):
    """
    Simple road simulator to test RL training.
    Draws a road with turns using OpenGL, and simulates
    basic differential-drive dynamics.
    """

    metadata = {"render.modes": ["human", "rgb_array", "app"], "video.frames_per_second": 30}

    cur_pos: np.ndarray # posición actual
    cam_offset: np.ndarray # distancia de la cámara
    road_tile_size: float # tamaño de las casillas de camino
    grid_width: int # ancho de las casillas
    grid_height: int # alto de las casillas
    step_count: int # contador de pasos
    timestamp: float # tiempo
    np_random: RandomState # randomización
    grid: List[TileDict] # diccionario de casillas

    def __init__(
        self,
        map_name: str = DEFAULT_MAP_NAME, # nombre del mapa
        max_steps: int = DEFAULT_MAX_STEPS, # número máximo de steps por episodio
        draw_curve: bool = False, # que se vean las curvas graficamente 
        draw_bbox: bool = False, # que se dibuje una caja alrededor del agente
        domain_rand: bool = True, # randomizacion del dominio
        frame_rate: float = DEFAULT_FRAMERATE, # cuadros 
        frame_skip: bool = DEFAULT_FRAME_SKIP, # saltar cuadros
        camera_width: int = DEFAULT_CAMERA_WIDTH, # ancho de la camara
        camera_height: int = DEFAULT_CAMERA_HEIGHT, # alto de la camara
        robot_speed: float = DEFAULT_ROBOT_SPEED, # velocidad del robot
        accept_start_angle_deg=DEFAULT_ACCEPT_START_ANGLE_DEG, # angulo de inicio aceptado
        full_transparency: bool = False,
        user_tile_start=None, # Definir casilla de inicio
        seed: int = None, # semilla
        distortion: bool = False,
        dynamics_rand: bool = False,
        camera_rand: bool = False,
        randomize_maps_on_reset: bool = False,
        num_tris_distractors: int = 12,
        color_ground: Sequence[float] = (0.15, 0.15, 0.15),
        color_sky: Sequence[float] = BLUE_SKY,
        style: str = "photos",
        enable_leds: bool = False,
        start_pose=None,
        goal=None #Poner una casilla de meta
    ):
        """

        :param map_name:
        :param max_steps:
        :param draw_curve:
        :param draw_bbox:
        :param domain_rand: If true, applies domain randomization
        :param frame_rate:
        :param frame_skip:
        :param camera_width:
        :param camera_height:
        :param robot_speed:
        :param accept_start_angle_deg:
        :param full_transparency:
        :param user_tile_start: If None, sample randomly. Otherwise (i,j). Overrides map start tile
        :param seed:
        :param distortion: If true, distorts the image with fish-eye approximation
        :param dynamics_rand: If true, perturbs the trim of the Duckiebot
        :param camera_rand: If true randomizes over camera miscalibration
        :param randomize_maps_on_reset: If true, randomizes the map on reset (Slows down training)
        :param style: String that represent which tiles will be loaded. One of ["photos", "synthetic"]
        :param enable_leds: Enables LEDs drawing.
        """
        self.enable_leds = enable_leds
        information = get_graphics_information()
        # logger.info(
        #     f"Information about the graphics card:",
        #     pyglet_version=pyglet.version,
        #     information=information,
        #     nvidia_around=os.path.exists("/proc/driver/nvidia/version"),
        # )

        # first initialize the RNG
        self.seed_value = seed
        self.seed(seed=self.seed_value)
        self.num_tris_distractors = num_tris_distractors
        self.color_ground = color_ground
        self.color_sky = list(color_sky)
        self.goal = goal
        # If true, then we publish all transparency information
        self.full_transparency = full_transparency

        # Map name, set in _load_map()
        self.map_name = None

        # Full map file path, set in _load_map()
        self.map_file_path = None

        # The parsed content of the map_file
        self.map_data = None

        # Maximum number of steps per episode
        self.max_steps = max_steps

        # Flag to draw the road curve
        self.draw_curve = draw_curve

        # Flag to draw bounding boxes
        self.draw_bbox = draw_bbox

        # Flag to enable/disable domain randomization
        self.domain_rand = domain_rand

        self.randomizer = Randomizer()

        # Frame rate to run at
        self.frame_rate = frame_rate
        self.delta_time = 1.0 / self.frame_rate

        # Number of frames to skip per action
        self.frame_skip = frame_skip

        # Produce graphical output
        self.graphics = True
        
        self.start_pose = start_pose
                
        # Set sensorr data shape
        # self.sensor_data_shape = (13,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Two-tuple of wheel torques, each in the range [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # self.action_space = spaces.Discrete(4)

        self.camera_width = camera_width
        self.camera_height = camera_height

        self.robot_speed = robot_speed
        self.reward_range = (-500, 500) # rango de recompensa

        # Window for displaying the environment to humans
        self.window = None

        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)

        # For displaying text
        self.text_label = pyglet.text.Label(font_name="Arial", font_size=14, x=5, y=WINDOW_HEIGHT - 19)

        # Create a frame buffer object for the observation
        # Contenedor de openGL se renderiza la imagen 
        self.multi_fbo, self.final_fbo = create_frame_buffers(self.camera_width, self.camera_height, 4)

        # Array to render the image into (for observation rendering)
        # Crea un array lleno de ceros con la forma de la observación que fue definida en la linea 324
        # Se crea para guardar las obersevaciones
        self.img_array = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
        # self.img_array = np.zeros(shape=self.observation_space.shape, dtype=np.float32)

        
        # print('-------------------------------------------')
        # print('-------------------------------------------')
        # print('-------------------------------------------')
        # print("img_array shape", self.img_array.shape)
        # # img_array shape (480, 640, 3)
        # print('-------------------------------------------')
        # print('----------------v---------------------------')
        # print('-------------------------------------------')
        

        # Create a frame buffer object for human rendering
        self.multi_fbo_human, self.final_fbo_human = create_frame_buffers(WINDOW_WIDTH, WINDOW_HEIGHT, 4)

        # Array to render the image into (for human rendering)
        self.img_array_human = np.zeros(shape=(WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        # self.img_array_human = np.zeros(shape=self.observation_space.shape, dtype=np.float32)


        # allowed angle in lane for starting position
        # angulo de inicio aceptado
        self.accept_start_angle_deg = accept_start_angle_deg

        # Load the map
        self._load_map(map_name)

        # Distortion params, if so, load the library, only if not bbox mode
        self.distortion = distortion and not draw_bbox
        self.camera_rand = False
        if not draw_bbox and distortion:
            if distortion:
                self.camera_rand = camera_rand

                self.camera_model = Distortion(camera_rand=self.camera_rand)

        # Used by the UndistortWrapper, always initialized to False
        self.undistort = False

        # Dynamics randomization
        self.dynamics_rand = dynamics_rand

        # Start tile
        # Indica la casilla de inicio
        self.user_tile_start = user_tile_start
        
        self.style = style

        # Cambiar el mapa al reset con un mapa aleatorio
        self.randomize_maps_on_reset = randomize_maps_on_reset

        # Busca en el directorio de mapas y los carga, solo prepara una lista de mapas con nombres
        if self.randomize_maps_on_reset:
            self.map_names = os.listdir(get_subdir_path("maps"))
            self.map_names = [
                _map for _map in self.map_names if not _map.startswith(("calibration", "regress"))
            ]
            self.map_names = [mapfile.replace(".yaml", "") for mapfile in self.map_names]

        # Initialize the state
        self.reset()

        # Inicializa la ultima accion como 00
        self.last_action = np.array([0, 0])
        
        # Inicializa la velocidad de las ruedas como 00
        self.wheelVels = np.array([0, 0])

    # Inicializa la parte del simulador que permite renderizar la carretera 
    def _init_vlists(self):

        ns = 8
        assert ns >= 2

        # half_size = self.road_tile_size / 2
        TS = self.road_tile_size

        def get_point(u, v):
            pu = u / (ns - 1)
            pv = v / (ns - 1)
            x = -TS / 2 + pu * TS
            z = -TS / 2 + pv * TS
            tu = pu
            tv = 1 - pv
            return (x, 0.0, z), (tu, tv)

        vertices = []
        textures = []
        normals = []
        colors = []
        for i, j in itertools.product(range(ns - 1), range(ns - 1)):
            tl_p, tl_t = get_point(i, j)
            tr_p, tr_t = get_point(i + 1, j)
            br_p, br_t = get_point(i, j + 1)
            bl_p, bl_t = get_point(i + 1, j + 1)
            normal = [0.0, 1.0, 0.0]

            color = (255, 255, 255, 255)
            vertices.extend(tl_p)
            textures.extend(tl_t)
            normals.extend(normal)
            colors.extend(color)

            vertices.extend(tr_p)
            textures.extend(tr_t)
            normals.extend(normal)
            colors.extend(color)

            vertices.extend(bl_p)
            textures.extend(bl_t)
            normals.extend(normal)
            colors.extend(color)

            vertices.extend(br_p)
            textures.extend(br_t)
            normals.extend(normal)
            colors.extend(color)

        total = len(vertices) // 3
        self.road_vlist = pyglet.graphics.vertex_list(
            total, ("v3f", vertices), ("t2f", textures), ("n3f", normals), ("c4B", colors)
        )
        logger.info("done")
        # Create the vertex list for the ground quad
        verts = [
            -1,
            -0.8,
            1,
            #
            -1,
            -0.8,
            -1,
            #
            1,
            -0.8,
            -1,  #
            1,
            -0.8,
            1,
        ]
        self.ground_vlist = pyglet.graphics.vertex_list(4, ("v3f", verts))

    # Define la función que hace reset al entorno   
    def reset(self, segment: bool = False):
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """

        # Step count since episode start
        # Poner a 0 el contador de pasos y el tiempo
        self.step_count = 0
        self.timestamp = 0.0

        # Robot's current speed
        # Poner a 0 la velocidad del robot !!!! IMPORTANTE!
        self.speed = 0.0


        # LLamar la lista de mapas que se crea anteriormente
        # Seleccionar y cargar un mapa aleatorio
        if self.randomize_maps_on_reset:
            map_name = self.np_random.choice(self.map_names)
            logger.info(f"Random map chosen: {map_name}")
            self._load_map(map_name)

        # Usar la conf de ramdomizer
        self.randomization_settings = self.randomizer.randomize(rng=self.np_random)

        # Horizon color
        # Note: we explicitly sample white and grey/black because
        # these colors are easily confused for road and lane markings
        
        # Si la randomizacion es true entonces se cambia el color del cielo
        if self.domain_rand:
            horz_mode = self.randomization_settings["horz_mode"]
            if horz_mode == 0:
                self.horizon_color = self._perturb(self.color_sky)
            elif horz_mode == 1:
                self.horizon_color = self._perturb(WALL_COLOR)
            elif horz_mode == 2: # gris oscuro 
                self.horizon_color = self._perturb([0.15, 0.15, 0.15], 0.4)
            elif horz_mode == 3: # gris claro
                self.horizon_color = self._perturb([0.9, 0.9, 0.9], 0.4)
        else:
            self.horizon_color = self.color_sky


        # Setup some basic lighting with a far away sun
        # Randomizar la posición de la luz si está activo 
        if self.domain_rand:
            light_pos = self.randomization_settings["light_pos"]
        else:
            # light_pos = [-40, 200, 100, 0.0]
            light_pos = [0.0, 3.0, 0.0, 1.0]

        # DIM = 0.0
        # Establecer la iluminación del entorno
        ambient = np.array([0.50 * DIM, 0.50 * DIM, 0.50 * DIM, 1])
        ambient = self._perturb(ambient, 0.3)
        diffuse = np.array([0.70 * DIM, 0.70 * DIM, 0.70 * DIM, 1])
        diffuse = self._perturb(diffuse, 0.99)
        # specular = np.array([0.3, 0.3, 0.3, 1])
        specular = np.array([0.0, 0.0, 0.0, 1])

        # logger.info(light_pos=light_pos, ambient=ambient, diffuse=diffuse, specular=specular)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*light_pos))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (gl.GLfloat * 4)(*ambient))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (gl.GLfloat * 4)(*diffuse))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, (gl.GLfloat * 4)(*specular))

        # gl.glLightfv(gl.GL_LIGHT0, gl.GL_CONSTANT_ATTENUATION, (gl.GLfloat * 1)(0.4))
        # gl.glLightfv(gl.GL_LIGHT0, gl.GL_LINEAR_ATTENUATION, (gl.GLfloat * 1)(0.3))
        # gl.glLightfv(gl.GL_LIGHT0, gl.GL_QUADRATIC_ATTENUATION, (gl.GLfloat * 1)(0.1))

        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_COLOR_MATERIAL)

        # Ground color
        # Color del fondo 
        self.ground_color = self._perturb(np.array(self.color_ground), 0.3)

        # Distance between the robot's wheels
        # Distancia entre las ruedas del robot
        self.wheel_dist = self._perturb(WHEEL_DIST)
        
        # Set default values

        # Distance bewteen camera and ground
        # Distancia entre la cámara y el suelo
        self.cam_height = CAMERA_FLOOR_DIST
        
        # Angle at which the camera is rotated
        # Angulo de rotación de la cámara
        self.cam_angle = [CAMERA_ANGLE, 0, 0]

        # Field of view angle of the camera
        # Distancia de campo de visión de la cámara
        self.cam_fov_y = CAMERA_FOV_Y

        # Perturb using randomization API (either if domain rand or only camera rand
        # Randomizar la camara y los valores como altura, angulo y campo de visión
        if self.domain_rand or self.camera_rand:
            self.cam_height *= self.randomization_settings["camera_height"]
            self.cam_angle = [CAMERA_ANGLE * self.randomization_settings["camera_angle"], 0, 0]
            self.cam_fov_y *= self.randomization_settings["camera_fov_y"]

        # Camera offset for use in free camera mode
        # Distancia de la camara y el agente 
        self.cam_offset = np.array([0, 0, 0])

        # Crea ruido en el piso del mapa, con vertices y triangulo que cambian de color y tamaño
        # Create the vertex list for the ground/noise triangles
        # These are distractors, junk on the floor
        numTris = self.num_tris_distractors # num de triangulos
        verts = [] # lista de vertices
        colors = [] # lista de colores
        
        # Se crean los vertices, por cada triangulo se crean 3 vertices, es decir se repite 3 veces por triangulo
        for _ in range(0, 3 * numTris):
            p = self.np_random.uniform(low=[-20, -0.6, -20], high=[20, -0.3, 20], size=(3,)) # se crea un punto aleatorio entre -0.6 y -0.3
            c = self.np_random.uniform(low=0, high=0.9) # genera un aleatorio entre 0 y 0.9 para el color
            c = self._perturb([c, c, c], 0.1)
            verts += [p[0], p[1], p[2]] # añade los vertices a las listas
            colors += [c[0], c[1], c[2]] # añade los colores a las listas

        # Se crea la lista de vertices en pyglet
        self.tri_vlist = pyglet.graphics.vertex_list(3 * numTris, ("v3f", verts), ("c3f", colors))

        # Randomize tile parameters
        # Randomizacion de casillas
        for tile in self.grid:
            
            # tomar el random que se define al inicio sino none
            rng = self.np_random if self.domain_rand else None

            # ver de que tipo es la casilla
            kind = tile["kind"]
            
            
            # fn = get_texture_file(f"tiles-processed/{self.style}/{kind}/texture")[0]
            # traer la textura de la casilla
            fn = get_texture_file(os.path.join("tiles-processed", self.style, kind, "texture"))[0]
            # ft = get_fancy_textures(self.style, texture_name)
            
            # cargar la textura
            t = load_texture(fn, segment=False, segment_into_color=False)
            
            # Crea la textura 
            tt = Texture(t, tex_name=kind, rng=rng)
            
            # Asigna la textura a la casilla
            tile["texture"] = tt

            # Random tile color multiplier
            # Perturba el color
            tile["color"] = self._perturb([1, 1, 1, 1], 0.2)

        # Randomize object parameters
        # Radomizar parametros de los objetos
        for obj in self.objects:
            # Randomize the object color
            # cambiar el color
            obj.color = self._perturb([1, 1, 1, 1], 0.3)

            # Randomize whether the object is visible or not
            # cambiar visibilidad
            if obj.optional and self.domain_rand:
                # obj.visible = self.np_random.integers(0, 2) == 0
                obj.visible = self.np_random.randint(0, 2) == 0
            else:
                obj.visible = True

        # If the map specifies a starting tile
        # Ubica al objeto si hay una casilla de inicio
        
        # Si la casilla está especificada en el inicio 
        if self.user_tile_start:
            # logger.info(f"using user tile start: {self.user_tile_start}")
            i, j = self.user_tile_start
            tile = self._get_tile(i, j)
            
            # si la casilla no existe 
            if tile is None:
                msg = "The tile specified does not exist."
                raise Exception(msg)
            # logger.debug(f"tile: {tile}")
            
            
        else:
            if self.start_tile is not None:
                tile = self.start_tile
            else:
                # Select a random drivable tile to start on
                # Revisa si hay casillas manejables
                if not self.drivable_tiles:
                    msg = "There are no drivable tiles. Use start_tile or self.user_tile_start"
                    raise Exception(msg)
                # tile_idx = self.np_random.integers(0, len(self.drivable_tiles))

                # Selecciona una casilla al azar dependiendo el numero de casillas manejables que tiene el mapa
                tile_idx = self.np_random.randint(0, len(self.drivable_tiles))                
                tile = self.drivable_tiles[tile_idx] # selecciona la casilla con el indice seleccionado
                
                # Que pasa acá? la casilla es conducible pero lo pone, puede que lo ponga en un lado del camino 
                # que no es la dirección correcta a la que debe ir
                
                # MAS ABAJO SE ESPEFICIFCA COMO PONER LA POSE INICIAL Y SÍ TIENE EN CUENTA LA DIRECCIÓN

        # If the map specifies a starting pose
        
        # configurando la posicion de inicio del agente en la celda seleccionada 
        if self.start_pose is not None:
            # logger.info(f"using map pose start: {self.start_pose}")

            # toma las cordenadas de la celda seleccionada
            i, j = tile["coords"]
            # print("i, j", i, j)
            # print("self.road_tile_size", self.road_tile_size)
            # print("self.start_pose", self.start_pose[0][2])
            
            # calcula x y z de acuerdo a la 
            
            x = i * self.road_tile_size + self.start_pose[0][0]
            z = j * self.road_tile_size + self.start_pose[0][2]
            propose_pos = np.array([x, 0, z])
            propose_angle = self.start_pose[1]

            # logger.info(f"Using map pose start. \n Pose: {propose_pos}, Angle: {propose_angle}")

        else:
            # Keep trying to find a valid spawn position on this tile
            for _ in range(MAX_SPAWN_ATTEMPTS):
                i, j = tile["coords"]

                # Choose a random position on this tile
                # Selecciona aleatoriamente una posición en la casilla 
                # Se usa el tamañao de la celda para hacerlo
                
                x = self.np_random.uniform(i, i + 1) * self.road_tile_size
                z = self.np_random.uniform(j, j + 1) * self.road_tile_size
                propose_pos = np.array([x, 0, z])

                # Choose a random direction
                # Selecciona un angulo aleatorio
                propose_angle = self.np_random.uniform(0, 2 * math.pi)

                # logger.debug('Sampled %s %s angle %s' % (propose_pos[0],
                #                                          propose_pos[1],
                #                                          np.rad2deg(propose_angle)))

                # If this is too close to an object or not a valid pose, retry
                inconvenient = self._inconvenient_spawn(propose_pos)

                if inconvenient:
                    # msg = 'The spawn was inconvenient.'
                    # logger.warning(msg)
                    continue

                invalid = not self._valid_pose(propose_pos, propose_angle, safety_factor=1.3)
                if invalid:
                    # msg = 'The spawn was invalid.'
                    # logger.warning(msg)
                    continue

                # If the angle is too far away from the driving direction, retry
                try:
                    lp = self.get_lane_pos2(propose_pos, propose_angle)
                except NotInLane:
                    continue
                M = self.accept_start_angle_deg
                ok = -M < lp.angle_deg < +M
                if not ok:
                    continue
                # Found a valid initial pose
                break
            else:
                msg = f"Could not find a valid starting pose after {MAX_SPAWN_ATTEMPTS} attempts"
                logger.warn(msg)
                propose_pos = np.array([1, 0, 1])
                propose_angle = 1

                # raise Exception(msg)

        self.cur_pos = propose_pos
        self.cur_angle = propose_angle

        # Velocidad inicial 
        init_vel = np.array([0, 0]) # IMPORTANTE

        # Initialize Dynamics model
        if self.dynamics_rand:
            trim = 0 + self.randomization_settings["trim"][0]
            p = get_DB18_uncalibrated(delay=0.15, trim=trim)
        else:
            p = get_DB18_nominal(delay=0.15)

        q = self.cartesian_from_weird(self.cur_pos, self.cur_angle)
        v0 = geometry.se2_from_linear_angular(init_vel, 0)
        c0 = q, v0
        self.state = p.initialize(c0=c0, t0=0)

        # logger.info(f"Starting at {self.cur_pos} {self.cur_angle}")
        # print('Apenas estoy acá')
        # Generate the first camera image
        # obs = self.render_obs(segment=segment)
        # print('Pase de la obs vieja')
        new_data = self.sensors_data()
        # print(f"New data: {new_data}")
        
        #         misc = self.get_agent_info()
        # print(f"New data (before conversion): {new_data}")


        # Return first observation
        return new_data

    def _load_map(self, map_name: str):
        """
        Load the map layout from a YAML file
        """
        # Carga el mapa según el nombre
        # Store the map name
        if os.path.exists(map_name) and os.path.isfile(map_name):
            # if env is loaded using gym's register function, we need to extract the map name from the complete url
            map_name = os.path.basename(map_name)
            assert map_name.endswith(".yaml")
            # Deja solo el nombre del mapa sin la extensión
            map_name = ".".join(map_name.split(".")[:-1])
        self.map_name = map_name

        # Get the full map file path
        self.map_file_path = get_resource_path(f"{map_name}.yaml")

        # logger.debug(f'loading map file "{self.map_file_path}"')

        # Abre el mapa y carga los obj
        with open(self.map_file_path, "r") as f:
            self.map_data = yaml.load(f, Loader=yaml.Loader)

        # Interpreta los obj del mapa
        self._interpret_map(self.map_data)

    def _interpret_map(self, map_data: MapFormat1):
        try:
            if not "tile_size" in map_data:
                msg = "Must now include explicit tile_size in the map data."
                raise InvalidMapException(msg)
            self.road_tile_size = map_data["tile_size"]
            self._init_vlists()

            tiles = map_data["tiles"]
            assert len(tiles) > 0
            assert len(tiles[0]) > 0

            # Create the grid
            self.grid_height = len(tiles)
            self.grid_width = len(tiles[0])
            # noinspection PyTypeChecker
            self.grid = [None] * self.grid_width * self.grid_height

            # We keep a separate list of drivable tiles
            self.drivable_tiles = []

            # For each row in the grid
            for j, row in enumerate(tiles):

                if len(row) != self.grid_width:
                    msg = "each row of tiles must have the same length"
                    raise InvalidMapException(msg, row=row)

                # For each tile in this row
                for i, tile in enumerate(row):
                    tile = tile.strip()

                    if tile == "empty":
                        continue

                    directions = ["S", "E", "N", "W"]
                    default_orient = "E"

                    if "/" in tile:
                        kind, orient = tile.split("/")
                        kind = kind.strip(" ")
                        orient = orient.strip(" ")
                        angle = directions.index(orient)

                    elif "4" in tile:
                        kind = "4way"
                        angle = directions.index(default_orient)

                    else:
                        kind = tile
                        angle = directions.index(default_orient)

                    DRIVABLE_TILES = [
                        "straight",
                        "curve_left",
                        "curve_right",
                        "3way_left",
                        "3way_right",
                        "4way",
                    ]
                    drivable = kind in DRIVABLE_TILES

                    # logger.info(f'kind {kind} drivable {drivable} row = {row}')

                    tile = cast(
                        TileDict, {"coords": (i, j), "kind": kind, "angle": angle, "drivable": drivable}
                    )

                    self._set_tile(i, j, tile)

                    if drivable:
                        tile["curves"] = self._get_curve(i, j)
                        self.drivable_tiles.append(tile)

            default_color = "red"

            self.mesh = get_duckiebot_mesh(default_color)
            self._load_objects(map_data)

            # Get the starting tile from the map, if specified
            self.start_tile = None
            if "start_tile" in map_data:
                coords = map_data["start_tile"]
                self.start_tile = self._get_tile(*coords)

            # Get the starting pose from the map, if specified
            # self.start_pose = None
            if "start_pose" in map_data:
                self.start_pose = map_data["start_pose"]
        except Exception as e:
            msg = "Cannot load map data"
            raise InvalidMapException(msg, map_data=map_data)

    def _load_objects(self, map_data: MapFormat1):
        # Create the objects array
        self.objects = []

        # The corners for every object, regardless if collidable or not
        self.object_corners = []

        # Arrays for checking collisions with N static objects
        # (Dynamic objects done separately)
        # (N x 2): Object position used in calculating reward
        self.collidable_centers = []

        # (N x 2 x 4): 4 corners - (x, z) - for object's boundbox
        self.collidable_corners = []

        # (N x 2 x 2): two 2D norms for each object (1 per axis of boundbox)
        self.collidable_norms = []

        # (N): Safety radius for object used in calculating reward
        self.collidable_safety_radii = []

        # For each object
        try:
            objects = map_data["objects"]
        except KeyError:
            pass
        else:
            if isinstance(objects, list):
                for obj_idx, desc in enumerate(objects):
                    kind = desc["kind"]
                    obj_name = f"ob{obj_idx:02d}-{kind}"
                    self.interpret_object(obj_name, desc)
            elif isinstance(objects, dict):
                for obj_name, desc in objects.items():
                    self.interpret_object(obj_name, desc)
            else:
                raise ValueError(objects)

        # If there are collidable objects
        if len(self.collidable_corners) > 0:
            self.collidable_corners = np.stack(self.collidable_corners, axis=0)
            self.collidable_norms = np.stack(self.collidable_norms, axis=0)

            # Stack doesn't do anything if there's only one object,
            # So we add an extra dimension to avoid shape errors later
            if len(self.collidable_corners.shape) == 2:
                self.collidable_corners = self.collidable_corners[np.newaxis]
                self.collidable_norms = self.collidable_norms[np.newaxis]

        self.collidable_centers = np.array(self.collidable_centers)
        self.collidable_safety_radii = np.array(self.collidable_safety_radii)

    def interpret_object(self, objname: str, desc: MapFormat1Object):
        kind = desc["kind"]

        W = self.grid_width
        tile_size = self.road_tile_size
        transform: SE2Transform = get_transform(desc, W, tile_size)
        # logger.info(desc=desc, transform=transform)

        pose = transform.as_SE2()

        pos, angle_rad = self.weird_from_cartesian(pose)

    
        # c = self.cartesian_from_weird(pos, angle_rad)
        # logger.debug(desc=desc, pose=geometry.SE2.friendly(pose), weird=(pos, angle_rad),
        # c=geometry.SE2.friendly(c))

        # pos = desc["pos"]
        # x, z = pos[0:2]
        # y = pos[2] if len(pos) == 3 else 0.0

        # rotate = desc.get("rotate", 0.0)
        optional = desc.get("optional", False)

        # pos = self.road_tile_size * np.array((x, y, z))

        # Load the mesh

        if kind == MapFormat1Constants.KIND_DUCKIEBOT:
            use_color = desc.get("color", "red")

            mesh = get_duckiebot_mesh(use_color)

        elif kind.startswith("sign"):
            change_materials: Dict[str, MatInfo]
            # logger.info(kind=kind, desc=desc)
            minfo = cast(MatInfo, {"map_Kd": f"{kind}.png"})
            change_materials = {"April_Tag": minfo}
            mesh = get_mesh("sign_generic", change_materials=change_materials)
        elif kind == "floor_tag":
            return
        else:
            mesh = get_mesh(kind)

        if "height" in desc:
            scale = desc["height"] / mesh.max_coords[1]
        else:
            if "scale" in desc:
                scale = desc["scale"]
            else:
                scale = 1.0
        assert not ("height" in desc and "scale" in desc), "cannot specify both height and scale"

        static = desc.get("static", True)
        # static = desc.get('static', False)
        # print('static is now', static)

        obj_desc = {
            "kind": kind,
            "mesh": mesh,
            "pos": pos,
            "angle": angle_rad,
            "scale": scale,
            "optional": optional,
            "static": static,
        }

        if static:
            if kind == MF1C.KIND_TRAFFICLIGHT:
                obj = TrafficLightObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
            else:
                obj = WorldObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
        else:
            if kind == MF1C.KIND_DUCKIEBOT:
                obj = DuckiebotObj(
                    obj_desc, self.domain_rand, SAFETY_RAD_MULT, WHEEL_DIST, ROBOT_WIDTH, ROBOT_LENGTH
                )
            elif kind == MF1C.KIND_DUCKIE:
                obj = DuckieObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT, self.road_tile_size)
            elif kind == MF1C.KIND_CHECKERBOARD:
                obj = CheckerboardObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT, self.road_tile_size)
            else:
                msg = "Object kind unknown."
                raise InvalidMapException(msg, kind=kind)

        self.objects.append(obj)

        # Compute collision detection information

        # angle = rotate * (math.pi / 180)

        # # Find drivable tiles object could intersect with
        # # possible_tiles = find_candidate_tiles(obj.obj_corners, self.road_tile_size)

        # If the object intersects with a drivable tile
        if (
            static
            and kind != MF1C.KIND_TRAFFICLIGHT
            # We want collision checking also for things outside the lanes
            # # and self._collidable_object(obj.obj_corners, obj.obj_norm, possible_tiles)
        ):
            # noinspection PyUnresolvedReferences
            self.collidable_centers.append(pos)  # XXX: changes types during initialization
            self.collidable_corners.append(obj.obj_corners.T)
            self.collidable_norms.append(obj.obj_norm)
            # noinspection PyUnresolvedReferences
            self.collidable_safety_radii.append(obj.safety_radius)  # XXX: changes types during initialization

    def close(self):
        pass

    def seed(self, seed=None):
        # Definir la semilla para el ramdomizer
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    # Coloca la baldosa en una casilla
    def _set_tile(self, i: int, j: int, tile: TileDict) -> None:
        assert 0 <= i < self.grid_width
        assert 0 <= j < self.grid_height
        index: int = j * self.grid_width + i
        self.grid[index] = tile

    # Obtiene la casilla en donde está el agente 
    def _get_tile(self, i: int, j: int) -> Optional[TileDict]:
        """
        Returns None if the duckiebot is not in a tile.
        """
        i = int(i)
        j = int(j)
        
        # print(i, j)
        if i < 0 or i >= self.grid_width:
            return None
        if j < 0 or j >= self.grid_height:
            return None
        
    #     print('-------------------------------------------')
    #     print(self.grid[j * self.grid_width + i])
        # print('Estoy en la casilla')
        # print(self.grid[j * self.grid_width + i])
        
        return self.grid[j * self.grid_width + i]
    

    def _perturb(self, val: Union[float, np.ndarray, List[float]], scale: float = 0.1) -> np.ndarray:
        """
        Add noise to a value. This is used for domain randomization.
        """
        assert 0 <= scale < 1

        val = np.array(val)

        if not self.domain_rand:
            return val

        if isinstance(val, np.ndarray):
            noise = self.np_random.uniform(low=1 - scale, high=1 + scale, size=val.shape)
            if val.size == 4:
                noise[3] = 1
        else:
            noise = self.np_random.uniform(low=1 - scale, high=1 + scale)

        res = val * noise

        return res

    # Verifica si un objeto es 
    def _collidable_object(self, obj_corners, obj_norm, possible_tiles):
        """
        A function to check if an object intersects with any
        drivable tiles, which would mean our agent could run into them.
        Helps optimize collision checking with agent during runtime
        """

        if possible_tiles.shape == 0:
            return False
        
        # crea una lista de casillas manejables
        drivable_tiles = []
        for c in possible_tiles:
            tile = self._get_tile(c[0], c[1])
            if tile and tile["drivable"]:
                drivable_tiles.append((c[0], c[1]))

        if not drivable_tiles:
            return False

        # Convierte la lista a un array
        drivable_tiles = np.array(drivable_tiles)

        # Tiles are axis aligned, so add normal vectors in bulk
        # Añade vectores normales a las casillas para ayudar a definir los limites
        tile_norms = np.array([[1, 0], [0, 1]] * len(drivable_tiles))

        # None of the candidate tiles are drivable, don't add object
        if len(drivable_tiles) == 0:
            return False

        # Find the corners for each candidate tile
        # Guarda las cordenadas de las esquinas de las casillas manejables 
        drivable_tiles = np.array(
            [
                tile_corners(self._get_tile(pt[0], pt[1])["coords"], self.road_tile_size).T
                for pt in drivable_tiles
            ]
        )

        # Stack doesn't do anything if there's only one object,
        # So we add an extra dimension to avoid shape errors later
        # Ajusta dimensiones
        if len(tile_norms.shape) == 2:
            tile_norms = tile_norms[np.newaxis]
        else:  # Stack works as expected
            drivable_tiles = np.stack(drivable_tiles, axis=0)
            tile_norms = np.stack(tile_norms, axis=0)

        # Only add it if one of the vertices is on a drivable tile
        # Revisa si alguna de las esquinas del obj se intersecta con las casillas manejables
        return intersects(obj_corners, drivable_tiles, obj_norm, tile_norms)

    def get_grid_coords(self, abs_pos: np.array) -> Tuple[int, int]:
        """
        Compute the tile indices (i,j) for a given (x,_,z) world position

        x-axis maps to increasing i indices
        z-axis maps to increasing j indices

        Note: may return coordinates outside of the grid if the
        position entered is outside of the grid.
        """
        # Tomar las coordenadas menos y porque siempre es 0
        x, _, z = abs_pos
        # Se calcula el indice de la dividiendo la posición entreel tamaño de la casilla
        i = math.floor(x / self.road_tile_size)
        j = math.floor(z / self.road_tile_size)
        
        # Convierte a entero y se regresa
        return int(i), int(j)

    def _get_curve(self, i, j):
        """
        Get the Bezier curve control points for a given tile
        """
        tile = self._get_tile(i, j)
        assert tile is not None

        kind = tile["kind"]
        angle = tile["angle"]

        # Each tile will have a unique set of control points,
        # Corresponding to each of its possible turns

        if kind.startswith("straight"):
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.25],
                            [-0.20, 0, 0.25],
                            [-0.20, 0, 0.50],
                        ],
                        [
                            [0.20, 0, 0.50],
                            [0.20, 0, 0.25],
                            [0.20, 0, -0.25],
                            [0.20, 0, -0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        elif kind == "curve_left":
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, 0.00],
                            [0.00, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [0.50, 0, -0.20],
                            [0.30, 0, -0.20],
                            [0.20, 0, -0.30],
                            [0.20, 0, -0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        elif kind == "curve_right":
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.20],
                            [-0.30, 0, -0.20],
                            [-0.50, 0, -0.20],
                        ],
                        [
                            [-0.50, 0, 0.20],
                            [-0.30, 0, 0.20],
                            [0.30, 0, 0.00],
                            [0.20, 0, -0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        # Hardcoded all curves for 3way intersection
        elif kind.startswith("3way"):
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.25],
                            [-0.20, 0, 0.25],
                            [-0.20, 0, 0.50],
                        ],
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, 0.00],
                            [0.00, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [0.20, 0, 0.50],
                            [0.20, 0, 0.25],
                            [0.20, 0, -0.25],
                            [0.20, 0, -0.50],
                        ],
                        [
                            [0.50, 0, -0.20],
                            [0.30, 0, -0.20],
                            [0.20, 0, -0.20],
                            [0.20, 0, -0.50],
                        ],
                        [
                            [0.20, 0, 0.50],
                            [0.20, 0, 0.20],
                            [0.30, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [0.50, 0, -0.20],
                            [0.30, 0, -0.20],
                            [-0.20, 0, 0.00],
                            [-0.20, 0, 0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        # Template for each side of 4way intersection
        elif kind.startswith("4way"):
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, 0.00],
                            [0.00, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.25],
                            [-0.20, 0, 0.25],
                            [-0.20, 0, 0.50],
                        ],
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.20],
                            [-0.30, 0, -0.20],
                            [-0.50, 0, -0.20],
                        ],
                    ]
                )
                * self.road_tile_size
            )
        else:
            msg = "Cannot get bezier for kind"
            raise InvalidMapException(msg, kind=kind)

        # Rotate and align each curve with its place in global frame
        if kind.startswith("4way"):
            fourway_pts = []
            # Generate all four sides' curves,
            # with 3-points template above
            for rot in np.arange(0, 4):
                mat = gen_rot_matrix(np.array([0, 1, 0]), rot * math.pi / 2)
                pts_new = np.matmul(pts, mat)
                pts_new += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])
                fourway_pts.append(pts_new)

            fourway_pts = np.reshape(np.array(fourway_pts), (12, 4, 3))
            return fourway_pts

        # Hardcoded each curve; just rotate and shift
        elif kind.startswith("3way"):
            threeway_pts = []
            mat = gen_rot_matrix(np.array([0, 1, 0]), angle * math.pi / 2)
            pts_new = np.matmul(pts, mat)
            pts_new += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])
            threeway_pts.append(pts_new)

            threeway_pts = np.array(threeway_pts)
            threeway_pts = np.reshape(threeway_pts, (6, 4, 3))
            return threeway_pts

        else:
            mat = gen_rot_matrix(np.array([0, 1, 0]), angle * math.pi / 2)
            pts = np.matmul(pts, mat)
            pts += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])

        # print('-------------------------------------------')
        # print(pts)
        return pts

    def closest_curve_point(
        self, pos: np.array, angle: float
    ) -> Tuple[Optional[np.array], Optional[np.array]]:
        """
        Get the closest point on the curve to a given point
        Also returns the tangent at that point.

        Returns None, None if not in a lane.
        """

        i, j = self.get_grid_coords(pos)
        tile = self._get_tile(i, j)

        if tile is None or not tile["drivable"]:
            return None, None

        # Find curve with largest dotproduct with heading
        curves = self._get_tile(i, j)["curves"]
        curve_headings = curves[:, -1, :] - curves[:, 0, :]
        curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
        dir_vec = get_dir_vec(angle)
        
        # print('buenas soy un vectorr')
        # print(dir_vec)

        dot_prods = np.dot(curve_headings, dir_vec)

        # Closest curve = one with largest dotprod
        cps = curves[np.argmax(dot_prods)]

        # Find closest point and tangent to this curve
        t = bezier_closest(cps, pos)
        point = bezier_point(cps, t)
        tangent = bezier_tangent(cps, t)
        
     
        
        
        return point, tangent

    def get_lane_pos2(self, pos, angle):
        """
        Get the position of the agent relative to the center of the right lane

        Raises NotInLane if the Duckiebot is not in a lane.
        """

        # Get the closest point along the right lane's Bezier curve,
        # and the tangent at that point
         
        # print('------AAAAAAAAAAAAAAAAAA-----------')
        point, tangent = self.closest_curve_point(pos, angle)
        
        # print(f'point {point}')
        # print(f'tangent {tangent}')
                
        if point is None or tangent is None:
            msg = f"Point not in lane: {pos}"
            raise NotInLane(msg)

        assert point is not None and tangent is not None

        # Compute the alignment of the agent direction with the curve tangent
        dirVec = get_dir_vec(angle)
        
        # print(f'dirvec {dirVec}')
        
        dotDir = np.dot(dirVec, tangent)
        
        # print(f'dotDir {dotDir}')
        dotDir = np.clip(dotDir, -1.0, +1.0)
        # print(f'dotDir clipped {dotDir}')

        # Compute the signed distance to the curve
        # Right of the curve is negative, left is positive
        
        # print(f'pos {pos}')
        
        posVec = pos - point
        # print(f'posVec {posVec}')
        
        upVec = np.array([0, 1, 0])
        
        
        rightVec = np.cross(tangent, upVec)
        # print(f'rightVec {rightVec}')
        
        signedDist = np.dot(posVec, rightVec)
        # print(f'signedDist {signedDist}')

        # Compute the signed angle between the direction and curve tangent
        # Right of the tangent is negative, left is positive
        angle_rad = math.acos(dotDir)
        # print(f'angle_rad {angle_rad}')

        # print('producto punto',np.dot(dirVec, rightVec))
        # print('dot dir',dotDir)
        if np.dot(dirVec, rightVec) < 0:
            angle_rad *= -1
            
  
        # solo lo cambia a grados
        angle_deg = np.rad2deg(angle_rad)
        # print('angle deg', angle_deg)
        
        # # return signedDist, dotDir, angle_deg
        # print('---------get lane pos----------------------')
        # print(f'angle',angle)
        # print(LanePosition(dist=signedDist, dot_dir=dotDir, angle_deg=angle_deg, angle_rad=angle_rad))
        
        
        return LanePosition(dist=signedDist, dot_dir=dotDir, angle_deg=angle_deg, angle_rad=angle_rad)

    def _drivable_pos(self, pos) -> bool:
        """
        Check that the given (x,y,z) position is on a drivable tile
        """

        coords = self.get_grid_coords(pos)
        tile = self._get_tile(*coords)
        if tile is None:
            msg = f"No tile found at {pos} {coords}"
            # logger.debug(msg)
            return False

        if not tile["drivable"]:
            msg = f"{pos} corresponds to tile at {coords} which is not drivable: {tile}"
            # logger.debug(msg)
            return False

        return True

    # Define la penalización que puede tener el agente cuando está cerca de un objeto
    # Entre más cerca esté, mayor será la penalización
    def proximity_penalty2(self, pos: g.T3value, angle: float) -> float:
        """
        Calculates a 'safe driving penalty' (used as negative rew.)
        as described in Issue #24

        Describes the amount of overlap between the "safety circles" (circles
        that extend further out than BBoxes, giving an earlier collision 'signal'
        The number is max(0, prox.penalty), where a lower (more negative) penalty
        means that more of the circles are overlapping
        """

        pos = _actual_center(pos, angle)
        if len(self.collidable_centers) == 0:
            static_dist = 0

        # Find safety penalty w.r.t static obstacles
        else:
            d = np.linalg.norm(self.collidable_centers - pos, axis=1)

            if not safety_circle_intersection(d, AGENT_SAFETY_RAD, self.collidable_safety_radii):
                static_dist = 0.0
            else:
                static_dist = safety_circle_overlap(d, AGENT_SAFETY_RAD, self.collidable_safety_radii)

        total_safety_pen = static_dist
        for obj in self.objects:
            # Find safety penalty w.r.t dynamic obstacles
            total_safety_pen += obj.proximity(pos, AGENT_SAFETY_RAD)
        
        # print('-------------------------------------------')
        # print('total_safety_pen', total_safety_pen)

        return total_safety_pen

    def _inconvenient_spawn(self, pos):
        """
        Check that agent spawn is not too close to any object
        """

        results = [
            np.linalg.norm(x.pos - pos) < max(x.max_coords) * 0.5 * x.scale + MIN_SPAWN_OBJ_DIST
            for x in self.objects
            if x.visible
        ]
        # print('-------------------------------------------')
        # print('results', results)
        return np.any(results)

    # Detecta si las esquinas de los obj y el agente se intersectan
    # si es así es porque hay colision
    def _collision(self, agent_corners):
        """
        Tensor-based OBB Collision detection
        """
        # Generate the norms corresponding to each face of BB
        agent_norm = generate_norm(agent_corners)

        # Check collisions with Static Objects
        if len(self.collidable_corners) > 0:
            collision = intersects(agent_corners, self.collidable_corners, agent_norm, self.collidable_norms)
            if collision:
                return True

        # Check collisions with Dynamic Objects
        for obj in self.objects:
            if obj.check_collision(agent_corners, agent_norm):
                return True

        # No collision with any object
        return False

    def _valid_pose(self, pos: g.T3value, angle: float, safety_factor: float = 1.0) -> bool:
        """
        Check that the agent is in a valid pose

        safety_factor = minimum distance
        """
        # print('-------------------------------------------')
        # print('pos', pos)
        # print('angle', angle)
        # print('safety_factor', safety_factor)
        # print('-------------------------------------------')
        # Compute the coordinates of the base of both wheels
        pos = _actual_center(pos, angle)
        f_vec = get_dir_vec(angle)
        r_vec = get_right_vec(angle)

        l_pos = pos - (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
        r_pos = pos + (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
        f_pos = pos + (safety_factor * 0.5 * ROBOT_LENGTH) * f_vec

        # Check that the center position and
        # both wheels are on drivable tiles and no collisions

        all_drivable = (
            self._drivable_pos(pos)
            and self._drivable_pos(l_pos)
            and self._drivable_pos(r_pos)
            and self._drivable_pos(f_pos)
        )

        # Recompute the bounding boxes (BB) for the agent
        agent_corners = get_agent_corners(pos, angle)
        no_collision = not self._collision(agent_corners)

        res = no_collision and all_drivable

        # if not res:
            # logger.debug(f"Invalid pose. Collision free: {no_collision} On drivable area: {all_drivable}")
            # logger.debug(f"safety_factor: {safety_factor}")
            # logger.debug(f"pos: {pos}")
            # logger.debug(f"l_pos: {l_pos}")
            # logger.debug(f"r_pos: {r_pos}")
            # logger.debug(f"f_pos: {f_pos}")

        return res

    def _check_intersection_static_obstacles(self, pos: g.T3value, angle: float) -> bool:
        agent_corners = get_agent_corners(pos, angle)
        agent_norm = generate_norm(agent_corners)
        # logger.debug(agent_corners=agent_corners, agent_norm=agent_norm)
        # Check collisions with Static Objects
        if len(self.collidable_corners) > 0:
            collision = intersects(agent_corners, self.collidable_corners, agent_norm, self.collidable_norms)
            if collision:
                return True
        return False

    cur_pose: np.ndarray
    cur_angle: float
    speed: float

    def update_physics(self, action, delta_time: float = None):
        # print("updating physics")
        if delta_time is None:
            delta_time = self.delta_time
            
            
        if np.any(self.wheelVels < 0):
            retroceso = -1
        else:
            retroceso = 1
            
            
        self.wheelVels = action * self.robot_speed * 1 ### IMPORTANTE VELOCIDAD
        # print(self.wheelVels)
        prev_pos = self.cur_pos

        # Update the robot's position
        self.cur_pos, self.cur_angle = _update_pos(self, action)
        
        
        # print(f'cur_angle: {self.cur_angle}')

        self.step_count += 1
        self.timestamp += delta_time

        self.last_action = action

        # Compute the robot's speed
        delta_pos = self.cur_pos - prev_pos
        
        
        # La velocidad del robot se calcula en base en 
        
        
        # Creé una variable retroceso para saber si el robot se está moviendo hacia adelante o hacia atrás
        self.speed = retroceso * np.linalg.norm(delta_pos) / delta_time # IMPORTANTE VELOCIDAD
        # print('speed', self.speed)
        # print('-------------------------------------------')
        # self.speed =  np.linalg.norm(delta_pos) / delta_time # IMPORTANTE VELOCIDAD
        
        # print('speed', self.speed)

        # print('-------------------------------------------')
        # print('action', action)
        # print('last_action', self.last_action)
        # print('cur_pos', self.cur_pos)
        # print('cur_angle', self.cur_angle)
        # # print
        # print('vel', self.speed)
        
        
        # Update world objects
        for obj in self.objects:
            if obj.kind == MapFormat1Constants.KIND_DUCKIEBOT:
                if not obj.static:
                    obj_i, obj_j = self.get_grid_coords(obj.pos)
                    same_tile_obj = [
                        o
                        for o in self.objects
                        if tuple(self.get_grid_coords(o.pos)) == (obj_i, obj_j) and o != obj
                    ]

                    obj.step_duckiebot(delta_time, self.closest_curve_point, same_tile_obj)
            else:
                # print("stepping all objects")
                obj.step(delta_time)

    # Devuelve la información del agente en un diccionario
    # La información es la posición, la velocidad, la penalización por proximidad
    def get_agent_info(self) -> dict:
        info = {}
        pos = self.cur_pos
        angle = self.cur_angle
        # Get the position relative to the right lane tangent

        info["action"] = list(self.last_action)
        if self.full_transparency:
            #             info['desc'] = """
            #
            # cur_pos, cur_angle ::  simulator frame (non cartesian)
            #
            # egovehicle_pose_cartesian :: cartesian frame
            #
            #     the map goes from (0,0) to (grid_height, grid_width)*self.road_tile_size
            #
            # """
            try:
                lp = self.get_lane_pos2(pos, angle)
                info["lane_position"] = lp.as_json_dict()
            except NotInLane:
                pass

            info["robot_speed"] = self.speed
            info["proximity_penalty"] = self.proximity_penalty2(pos, angle)
            info["cur_pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]
            info["cur_angle"] = float(angle)
            info["wheel_velocities"] = [self.wheelVels[0], self.wheelVels[1]]

            # put in cartesian coordinates
            # (0,0 is bottom left)
            # q = self.cartesian_from_weird(self.cur_pos, self.)
            # info['cur_pos_cartesian'] = [float(p[0]), float(p[1])]
            # info['egovehicle_pose_cartesian'] = {'~SE2Transform': {'p': [float(p[0]), float(p[1])],
            #                                                        'theta': angle}}

            info["timestamp"] = self.timestamp
            info["tile_coords"] = list(self.get_grid_coords(pos))
            # info['map_data'] = self.map_data
            
        # print('--------------------MISC-----------------------')
        # print('info', info)

        misc = {}
        misc["Simulator"] = info
        
        # print(misc)
        
        return misc
    
    
    def sensors_data(self):
        pos = self.cur_pos
        angle = self.cur_angle

        try:
            lp = self.get_lane_pos2(pos, angle)
        except NotInLane:
            pass
       
               
        dats = []
        dats.append(self.speed)
        dats.append(self.proximity_penalty2(pos, angle))
        dats.append(lp.dist)
        # dats.append(lp.dot_dir)
        dats.append(angle) 
        dats.append(lp.angle_deg) # con respecto al carril 
        
        
        return self.speed, self.proximity_penalty2(pos, angle), lp.dist, lp.dot_dir # lp. dot_dir
    
    
    def transform_angle(self, dot_dir):
        """
        Transforma el ángulo para que los valores cercanos a la línea del eje X (180° o -180°) sean negativos,
        y los valores cercanos a la dirección correcta (0°) sean positivos.
        """
        # Asegurarse de que el ángulo esté en el rango [-180, 180]
        angle = dot_dir
        
        # Convertir el ángulo a un valor negativo si está en los cuadrantes "al revés"
        if angle > 0.45:
            return dot_dir # Mueve el ángulo a un valor negativo (-180° a -90°)
        elif angle < 0.45:
            return dot_dir * -1  # Mueve el ángulo a un valor negativo (-180° a -90°)
        else:
            return angle  # El ángulo ya está entre [-90°, 90°] y lo dejamos igual


  

    def cartesian_from_weird(self, pos, angle) -> np.ndarray:
        gx, gy, gz = pos
        grid_height = self.grid_height
        tile_size = self.road_tile_size

        # this was before but obviously doesn't work for grid_height = 1
        # cp = [gx, (grid_height - 1) * tile_size - gz]
        cp = [gx, grid_height * tile_size - gz]

        # print('-------------------------------------------')
        # print(geometry.SE2_from_translation_angle(np.array(cp), angle))
        # print('pos:', pos)
        # print('angle:', angle)
        
        return geometry.SE2_from_translation_angle(np.array(cp), angle)

    def weird_from_cartesian(self, q: SE2value) -> Tuple[list, float]:

        cp, angle = geometry.translation_angle_from_SE2(q)

        gx = cp[0]
        gy = 0
        # cp[1] = (grid_height - 1) * tile_size - gz
        GH = self.grid_height
        tile_size = self.road_tile_size
        # this was before but obviously doesn't work for grid_height = 1
        # gz = (grid_height - 1) * tile_size - cp[1]
        gz = GH * tile_size - cp[1]
        return [gx, gy, gz], angle


    import numpy as np
    def asignar_signo_por_cuadrante(self, angulo, deg):
        if deg <= -60 and angulo >= 50:
            # Cuadrante 1: Positivo (Dirección Correcta)
            return deg * -1
        elif deg >= 60 and angulo >= -50:
            # Cuadrante 2: Positivo (Dirección Correcta)
            return deg * -1
        else:
            return deg

    def transformar_a_360(self, angulo):
        
        angulonuevo = angulo
        multiplicador = 1
       
        if angulo >= 0:
            # Los ángulos positivos ya están en el rango correcto (0° a 180°)
            angulonuevo = angulo
        else:
            # Los ángulos negativos se transforman sumándoles 360° (0° a 360°)
            angulonuevo =  360 + angulo
            
        if 60 < angulonuevo < 300:
            multiplicador = -1
        
        
        return multiplicador
            
        

    def get_dist_goal(self, pos):
        # Obtén las coordenadas de la meta
        i, j = self.goal
        
        # Calcula las coordenadas en el espacio del mapa
        x_goal = i * self.road_tile_size
        z_goal = j * self.road_tile_size

        # Coordenadas del vehículo
        x_car, _, z_car = pos

        # Calcula la distancia euclidiana entre el vehículo y la meta
        dist = np.sqrt((x_goal - x_car) ** 2 + (z_goal - z_car) ** 2)

        # Imprime resultados para debug
        # print('-------------------------------------------')
        # print(f'Coordenadas de la meta: ({x_goal}, {z_goal})')
        # print(f'Coordenadas del vehículo: ({x_car}, {z_car})')
        # print(f'Distancia a la meta: {dist}')
    
        return dist

        
    def compute_reward(self, pos, angle, speed):
        # Compute the collision avoidance penalty
        col_penalty = self.proximity_penalty2(pos, angle)
        
        # Get the position relative to the right lane tangent
        try:
            lp = self.get_lane_pos2(pos, angle)
        except NotInLane:
            reward = 40 * col_penalty
        else:
            # Calcula la distancia a la meta
            dist_goal = self.get_dist_goal(pos) 
       
       
            multipl = self.transformar_a_360(np.rad2deg(self.cur_angle))
            
            velocidad = 20 * speed
            direccion = lp.dot_dir * multipl
            
            distancia = -10 * np.abs(lp.dist)
            penalizacion =  (40 * col_penalty)
            
            
            if distancia > -0.75:
                distancia = distancia
            else:
                distancia = distancia * 5
            
        
            
            if velocidad < 0:
                if direccion < 0:
                    velocidad = velocidad * direccion * -1  # Esto asegura que el resultado sea negativo.
                else:
                    velocidad = velocidad * direccion  # Esto mantendrá la velocidad negativa.
            else:
                velocidad = velocidad * direccion  # Esto maneja el caso cuando velocidad es positiva.
            
          
            reward = (velocidad * direccion  + distancia + penalizacion - dist_goal)
        
        return reward
  
        
    

    def step(self, action: np.ndarray):
        # Mantener las acciones en el rango [-1, 1]
        action = np.clip(action, -1, 1)
        
        # Actions could be a Python list
        # Convertir a un array
        action = np.array(action)
        
        
        # print('-----------------ACCION-------------------')
        # print(action)
        
        # Se actualiza la fisica del entorno que invluye la posición, el ángulo y la velocidad
        for _ in range(self.frame_skip):
            self.update_physics(action)
            
            
        # Genera una observación del entorno como imagen
        # Obtiene información del agente
        # Generate the current camera image
        # obs = self.render_obs()
        # print('Buenas tardes')
        dats = self.sensors_data()
        misc = self.get_agent_info()
        # print('-------------------------------------------')
        # print('dats', dats)
        # print('misc', misc)
        
        d = self._compute_done_reward()
        misc["Simulator"]["msg"] = d.done_why

        # print('-------------------------------------------')
        # print('obs', obs)
        # print('reward', d.reward)
        # print(action)
        # print('#############################################')

        return dats, d.reward, d.done, misc

    # Revisa si se cumplió alguna de las condiciones de terminación por episodio
    # Primmero con la pose, después con el conteo de pasos y si se continua o no 
    def _compute_done_reward(self) -> DoneRewardInfo:
        # If the agent is not in a valid pose (on drivable tiles)
        
        
        # print('-------------------------------------------')
        
        coords = self.get_grid_coords(self.cur_pos)
        tile = self._get_tile(*coords)
        meta = np.array(self.goal)
        agente = np.array(tile["coords"])
        
        
        if not self._valid_pose(self.cur_pos, self.cur_angle):
            msg = "Stopping the simulator because we are at an invalid pose."
            # logger.info(msg)
            reward = REWARD_INVALID_POSE
            done_code = "invalid-pose"
            done = True
        # If the maximum time step count is reached
        elif self.step_count >= self.max_steps:
            msg = "Stopping the simulator because we reached max_steps = %s" % self.max_steps
            logger.info(msg)
            done = True
            reward = 0
            done_code = "max-steps-reached"
        elif np.array_equal(meta, agente):
            msg = "Stopping the simulator because we reached the goal."
            logger.info(msg)
            done = True
            reward = REWARD_SUCCESS
            done_code = "goal-reached"
        else:
            done = False
            reward = self.compute_reward(self.cur_pos, self.cur_angle, self.speed)
            msg = ""
            done_code = "in-progress"
        return DoneRewardInfo(done=done, done_why=msg, reward=reward, done_code=done_code)

    def _render_img(
        self,
        width: int,
        height: int,
        multi_fbo,
        final_fbo,
        img_array,
        top_down: bool = True,
        segment: bool = False,
    ) -> np.ndarray:
        """
        Render an image of the environment into a frame buffer
        Produce a numpy RGB array image as output
        """

        if not self.graphics:
            return np.zeros((height, width, 3), np.uint8)

        # Switch to the default context
        # This is necessary on Linux nvidia drivers
        # pyglet.gl._shadow_window.switch_to()
        self.shadow_window.switch_to()

        if segment:
            gl.glDisable(gl.GL_LIGHT0)
            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_COLOR_MATERIAL)
        else:
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_COLOR_MATERIAL)

        # note by default the ambient light is 0.2,0.2,0.2
        # ambient = [0.03, 0.03, 0.03, 1.0]
        ambient = [0.3, 0.3, 0.3, 1.0]

        gl.glEnable(gl.GL_POLYGON_SMOOTH)

        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, (gl.GLfloat * 4)(*ambient))
        # Bind the multisampled frame buffer
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, multi_fbo)
        gl.glViewport(0, 0, width, height)

        # Clear the color and depth buffers

        c0, c1, c2 = self.horizon_color if not segment else [255, 0, 255]
        gl.glClearColor(c0, c1, c2, 1.0)
        gl.glClearDepth(1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(self.cam_fov_y, width / float(height), 0.04, 100.0)

        # Set modelview matrix
        # Note: we add a bit of noise to the camera position for data augmentation
        pos = self.cur_pos
        angle = self.cur_angle
        # logger.info('Pos: %s angle %s' % (self.cur_pos, self.cur_angle))
        if self.domain_rand:
            pos = pos + self.randomization_settings["camera_noise"]

        x, y, z = pos + self.cam_offset
        dx, dy, dz = get_dir_vec(angle)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        if self.draw_bbox:
            y += 0.8
            gl.glRotatef(90, 1, 0, 0)
        elif not top_down:
            y += self.cam_height
            gl.glRotatef(self.cam_angle[0], 1, 0, 0)
            gl.glRotatef(self.cam_angle[1], 0, 1, 0)
            gl.glRotatef(self.cam_angle[2], 0, 0, 1)
            gl.glTranslatef(0, 0, CAMERA_FORWARD_DIST)

        if top_down:
            a = (self.grid_width * self.road_tile_size) / 2
            b = (self.grid_height * self.road_tile_size) / 2
            fov_y_deg = self.cam_fov_y
            fov_y_rad = np.deg2rad(fov_y_deg)
            H_to_fit = max(a, b) + 0.1  # borders

            H_FROM_FLOOR = H_to_fit / (np.tan(fov_y_rad / 2))

            look_from = a, H_FROM_FLOOR, b
            look_at = a, 0.0, b - 0.01
            up_vector = 0.0, 1.0, 0
            gl.gluLookAt(*look_from, *look_at, *up_vector)
        else:
            look_from = x, y, z
            look_at = x + dx, y + dy, z + dz
            up_vector = 0.0, 1.0, 0.0
            gl.gluLookAt(*look_from, *look_at, *up_vector)

        # Draw the ground quad
        gl.glDisable(gl.GL_TEXTURE_2D)
        # background is magenta when segmenting for easy isolation of main map image
        gl.glColor3f(*self.ground_color if not segment else [255, 0, 255])  # XXX
        gl.glPushMatrix()
        gl.glScalef(50, 0.01, 50)
        self.ground_vlist.draw(gl.GL_QUADS)
        gl.glPopMatrix()

        # Draw the ground/noise triangles
        if not segment:
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.1, 0.0)
            self.tri_vlist.draw(gl.GL_TRIANGLES)
            gl.glPopMatrix()

        # Draw the road quads
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        add_lights = False
        if add_lights:
            for i in range(1):
                li = gl.GL_LIGHT0 + 1 + i
                # li_pos = [i + 1, 1, i + 1, 1]

                li_pos = [0.0, 0.2, 3.0, 1.0]
                diffuse = [0.0, 0.0, 1.0, 1.0] if i % 2 == 0 else [1.0, 0.0, 0.0, 1.0]
                ambient = [0.0, 0.0, 0.0, 1.0]
                specular = [0.0, 0.0, 0.0, 1.0]
                spot_direction = [0.0, -1.0, 0.0]
                logger.debug(
                    li=li, li_pos=li_pos, ambient=ambient, diffuse=diffuse, spot_direction=spot_direction
                )
                gl.glLightfv(li, gl.GL_POSITION, (gl.GLfloat * 4)(*li_pos))
                gl.glLightfv(li, gl.GL_AMBIENT, (gl.GLfloat * 4)(*ambient))
                gl.glLightfv(li, gl.GL_DIFFUSE, (gl.GLfloat * 4)(*diffuse))
                gl.glLightfv(li, gl.GL_SPECULAR, (gl.GLfloat * 4)(*specular))
                gl.glLightfv(li, gl.GL_SPOT_DIRECTION, (gl.GLfloat * 3)(*spot_direction))
                # gl.glLightfv(li, gl.GL_SPOT_EXPONENT, (gl.GLfloat * 1)(64.0))
                gl.glLightf(li, gl.GL_SPOT_CUTOFF, 60)

                gl.glLightfv(li, gl.GL_CONSTANT_ATTENUATION, (gl.GLfloat * 1)(1.0))
                # gl.glLightfv(li, gl.GL_LINEAR_ATTENUATION, (gl.GLfloat * 1)(0.1))
                gl.glLightfv(li, gl.GL_QUADRATIC_ATTENUATION, (gl.GLfloat * 1)(0.2))
                gl.glEnable(li)

        # For each grid tile
        for i, j in itertools.product(range(self.grid_width), range(self.grid_height)):

            # Get the tile type and angle
            tile = self._get_tile(i, j)

            if tile is None:
                continue

            # kind = tile['kind']
            angle = tile["angle"]
            color = tile["color"]
            texture = tile["texture"]

            # logger.info('drawing', tile_color=color)

            gl.glColor4f(*color)

            gl.glPushMatrix()
            TS = self.road_tile_size
            gl.glTranslatef((i + 0.5) * TS, 0, (j + 0.5) * TS)
            gl.glRotatef(angle * 90 + 180, 0, 1, 0)

            # gl.glEnable(gl.GL_BLEND)
            # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            # Bind the appropriate texture
            texture.bind(segment)

            self.road_vlist.draw(gl.GL_QUADS)
            # gl.glDisable(gl.GL_BLEND)

            gl.glPopMatrix()

            if self.draw_curve and tile["drivable"]:
                # Find curve with largest dotproduct with heading
                curves = tile["curves"]
                curve_headings = curves[:, -1, :] - curves[:, 0, :]
                curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
                dirVec = get_dir_vec(angle)
                dot_prods = np.dot(curve_headings, dirVec)

                # Current ("closest") curve drawn in Red
                pts = curves[np.argmax(dot_prods)]
                bezier_draw(pts, n=20, red=True)

                pts = self._get_curve(i, j)
                for idx, pt in enumerate(pts):
                    # Don't draw current curve in blue
                    if idx == np.argmax(dot_prods):
                        continue
                    bezier_draw(pt, n=20)

        # For each object
        for obj in self.objects:
            obj.render(draw_bbox=self.draw_bbox, segment=segment, enable_leds=self.enable_leds)

        # Draw the agent's own bounding box
        if self.draw_bbox:
            corners = get_agent_corners(pos, angle)
            gl.glColor3f(1, 0, 0)
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex3f(corners[0, 0], 0.01, corners[0, 1])
            gl.glVertex3f(corners[1, 0], 0.01, corners[1, 1])
            gl.glVertex3f(corners[2, 0], 0.01, corners[2, 1])
            gl.glVertex3f(corners[3, 0], 0.01, corners[3, 1])
            gl.glEnd()

        if top_down:
            gl.glPushMatrix()
            gl.glTranslatef(*self.cur_pos)
            gl.glScalef(1, 1, 1)
            gl.glRotatef(self.cur_angle * 180 / np.pi, 0, 1, 0)
            # glColor3f(*self.color)
            self.mesh.render()
            gl.glPopMatrix()
        draw_xyz_axes = False
        if draw_xyz_axes:
            draw_axes()
        # Resolve the multisampled frame buffer into the final frame buffer
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, multi_fbo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, final_fbo)
        gl.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)

        # Copy the frame buffer contents into a numpy array
        # Note: glReadPixels reads starting from the lower left corner
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, final_fbo)
        gl.glReadPixels(
            0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_array.ctypes.data_as(POINTER(gl.GLubyte))
        )

        # Unbind the frame buffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Flip the image because OpenGL maps (0,0) to the lower-left corner
        # Note: this is necessary for gym.wrappers.Monitor to record videos
        # properly, otherwise they are vertically inverted.
        img_array = np.ascontiguousarray(np.flip(img_array, axis=0))

        return img_array

    def render_obs(self, segment: bool = False) -> np.ndarray:
        """
        Render an observation from the point of view of the agent
        """

        observation = self._render_img(
            self.camera_width,
            self.camera_height,
            self.multi_fbo,
            self.final_fbo,
            self.img_array,
            top_down=False,
            segment=segment,
        )

        # self.undistort - for UndistortWrapper
        if self.distortion and not self.undistort:
            observation = self.camera_model.distort(observation)

        return observation

    def render(self, mode: str = "human", close: bool = False, segment: bool = False):
        """
        Render the environment for human viewing

        mode: "human", "top_down", "free_cam", "rgb_array"

        """
        assert mode in ["human", "top_down", "free_cam", "rgb_array"]

        if close:
            if self.window:
                self.window.close()
            return

        top_down = mode == "top_down"
        # Render the image
        img = self._render_img(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            self.multi_fbo_human,
            self.final_fbo_human,
            self.img_array_human,
            top_down=top_down,
            segment=segment,
        )

        # self.undistort - for UndistortWrapper
        if self.distortion and not self.undistort and mode != "free_cam":
            img = self.camera_model.distort(img)

        if mode == "rgb_array":
            return img

        if self.window is None:
            config = gl.Config(double_buffer=False)
            self.window = window.Window(
                width=WINDOW_WIDTH, height=WINDOW_HEIGHT, resizable=False, config=config
            )

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # Bind the default frame buffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Setup orghogonal projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 10)

        # Draw the image to the rendering window
        width = img.shape[1]
        height = img.shape[0]
        img = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = image.ImageData(
            width,
            height,
            "RGB",
            img.ctypes.data_as(POINTER(gl.GLubyte)),
            pitch=width * 3,
        )
        img_data.blit(0, 0, 0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

        # Display position/state information
        if mode != "free_cam":
            x, y, z = self.cur_pos
            self.text_label.text = (
                f"pos: ({x:.2f}, {y:.2f}, {z:.2f}), angle: "
                f"{np.rad2deg(self.cur_angle):.1f} deg, steps: {self.step_count}, "
                f"speed: {self.speed:.2f} m/s"
                f"tiempo: {self.speed:.2f} m/s"
                
            )
            self.text_label.draw()

        # Force execution of queued commands
        gl.glFlush()

        return img


def get_dir_vec(cur_angle: float) -> np.ndarray:
    """
    Vector pointing in the direction the agent is looking
    """

    x = math.cos(cur_angle)
    z = -math.sin(cur_angle)
    
    # print('---------------get_dir_vec----------------------')
    # print('x', x)
    # print('z', z)
    return np.array([x, 0, z])


def get_right_vec(cur_angle: float) -> np.ndarray:
    """
    Vector pointing to the right of the agent
    """

    x = math.sin(cur_angle)
    z = math.cos(cur_angle)
    return np.array([x, 0, z])


def _update_pos(self, action):
    """
    Update the position of the robot, simulating differential drive

    returns pos, angle
    """

    action = DynamicsInfo(motor_left=action[0], motor_right=action[1])
    self.state = self.state.integrate(self.delta_time, action)
    q = self.state.TSE2_from_state()[0]
    pos, angle = self.weird_from_cartesian(q)
    pos = np.asarray(pos)
    return pos, angle


def get_duckiebot_mesh(color: str) -> ObjMesh:
    change_materials: Dict[str, MatInfo]

    color = np.array(get_duckiebot_color_from_colorname(color))[:3]
    change_materials = {
        "gkmodel0_chassis_geom0_mat_001-material": {"Kd": color},
        "gkmodel0_chassis_geom0_mat_001-material.001": {"Kd": color},
    }
    return get_mesh("duckiebot", change_materials=change_materials)


def _actual_center(pos, angle):
    """
    Calculate the position of the geometric center of the agent
    The value of self.cur_pos is the center of rotation.
    """

    dir_vec = get_dir_vec(angle)
    return pos + (CAMERA_FORWARD_DIST - (ROBOT_LENGTH / 2)) * dir_vec


def get_agent_corners(pos, angle):
    agent_corners = agent_boundbox(
        _actual_center(pos, angle), ROBOT_WIDTH, ROBOT_LENGTH, get_dir_vec(angle), get_right_vec(angle)
    )
    return agent_corners


class FrameBufferMemory:
    multi_fbo: int
    final_fbo: int
    img_array: np.ndarray
    width: int

    height: int

    def __init__(self, *, width: int, height: int):
        """H, W"""
        self.width = width
        self.height = height

        # that's right, it's inverted
        self.multi_fbo, self.final_fbo = create_frame_buffers(width, height, 4)
        self.img_array = np.zeros(shape=(height, width, 3), dtype=np.uint8)


def draw_axes():
    gl.glPushMatrix()
    gl.glLineWidth(4.0)
    gl.glTranslatef(0.0, 0.0, 0.0)

    gl.glBegin(gl.GL_LINES)
    L = 0.3
    gl.glColor3f(1.0, 0.0, 0.0)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(L, 0.0, 0.0)

    gl.glColor3f(0.0, 1.0, 0.0)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, L, 0.0)

    gl.glColor3f(0.0, 0.0, 1.0)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, 0.0, L)
    gl.glEnd()

    gl.glPopMatrix()


