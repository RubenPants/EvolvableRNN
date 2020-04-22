"""
env_visualizing_live.py

The visualizer gives a live visualization of a bot's run.
"""
import numpy as np
import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions

from config import Config
from environment.game import Game, get_game
from population.population import Population
from population.utils.network_util.feed_forward_net import make_net
from utils.dictionary import D_DONE, D_SENSOR_LIST


class LiveVisualizer:
    """
    The visualizer will visualize the run of a single genome from the population in a game of choice. This is done by
    the use of pymunk.
    """
    
    __slots__ = {
        'speedup', 'state', 'finished', 'time', 'score', 'p2m',
        'make_net', 'pop_config', 'game_config',
        'debug',
    }
    
    def __init__(self,
                 pop: Population,
                 game_config: Config,
                 debug: bool = True,
                 speedup: float = 3):
        """
        The visualizer provides methods used to visualize the performance of a single genome.
        
        :param pop: Population object
        :param game_config: Config file for current game-configurations
        :param debug: Generates prints (CLI) during visualization
        :param speedup: Specifies the relative speedup the virtual environment faces towards the real world
        """
        # Visualizer specific parameters
        self.speedup = speedup
        self.state = None
        self.finished = False
        self.time = 0
        self.score = 0
        self.p2m = 0
        
        # Network specific parameters
        self.pop_config = pop.config
        self.game_config = game_config
        
        # Debug options
        self.debug = debug
    
    def visualize(self, genome, game_id: int):
        """
        Visualize the performance of a single genome.
        
        :param genome: Tuple (genome_id, genome_class)
        :param game_id: ID of the game that will be used for evaluation
        """
        # Create the requested game
        game: Game = get_game(game_id, cfg=self.game_config)
        self.p2m = game.game_config.p2m
        
        # Create space in which game will be played
        window = pyglet.window.Window(game.x_axis * self.p2m,
                                      game.y_axis * self.p2m,
                                      "Robot Simulator - Game {id:03d}".format(id=game_id),
                                      resizable=False,
                                      visible=True)
        window.set_location(100, 100)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        
        # Setup the requested game
        self.state = game.reset()[D_SENSOR_LIST]
        self.finished = False
        self.score = 0
        
        # Make the network used during visualization
        net = make_net(genome=genome,
                       genome_config=self.pop_config.genome,
                       batch_size=1,
                       initial_read=self.state,
                       )
        
        # Create the visualize-environment
        space = pymunk.Space()
        options = DrawOptions()
        
        # Draw static objects - walls
        if game.wall_bound:
            x_axis = game.x_axis
            y_axis = game.y_axis
            corners = [(0, 0), (0, y_axis * self.p2m), (x_axis * self.p2m, y_axis * self.p2m), (x_axis * self.p2m, 0)]
            for c in range(4):
                wall_shape = pymunk.Segment(space.static_body,
                                            a=corners[c],
                                            b=corners[(c + 1) % 4],
                                            radius=0.1 * self.p2m)  # 5cm walls
                wall_shape.color = (0, 0, 0)
                space.add(wall_shape)
        
        # Draw static objects - target
        target_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        target_body.position = game.target * self.p2m
        target_shape = pymunk.Circle(body=target_body,
                                     radius=game.bot_config.radius * self.p2m)
        target_shape.sensor = True
        target_shape.color = (0, 128, 0)
        space.add(target_body, target_shape)
        
        # Init player²
        m = pymunk.moment_for_circle(mass=2,
                                     inner_radius=0,
                                     outer_radius=game.bot_config.radius * self.p2m)
        player_body = pymunk.Body(mass=1, moment=m)
        player_body.position = game.player.pos * self.p2m
        player_body.angle = game.player.angle
        player_shape = pymunk.Circle(body=player_body,
                                     radius=game.bot_config.radius * self.p2m)
        player_shape.color = (255, 0, 0)
        space.add(player_body, player_shape)
        label = pyglet.text.Label(f'{self.time}',
                                  font_size=16,
                                  color=(100, 100, 100, 100),
                                  x=window.width - 20, y=window.height - 20,
                                  anchor_x='center', anchor_y='center')
        
        @window.event
        def on_draw():
            window.clear()
            label.draw()
            space.debug_draw(options=options)
        
        def update_method(_):  # Input dt ignored
            dt = 1 / game.game_config.fps
            self.time += dt
            label.text = str(int(self.time))
            
            # Stop when target is reached
            if not self.finished:
                # Query the game for the next action
                action = net(np.asarray([self.state]))
                if self.debug:
                    print(f"Passed time: {round(dt, 3)}")
                    print(f"Location: x={round(player_body.position.x / self.p2m, 2)}, "
                          f"y={round(player_body.position.y / self.p2m, 2)}")
                    print(f"Orientation: {round(player_body.angle, 2)}")
                    print("Action: lw={l}, rw={r}".format(l=round(action[0][0], 3), r=round(action[0][1], 3)))
                    print("Observation:", [round(s, 3) for s in self.state])
                
                # Progress game by one step
                obs = game.step_dt(dt=dt, l=action[0][0], r=action[0][1])
                self.finished = obs[D_DONE]
                self.state = obs[D_SENSOR_LIST]
                
                # Update space's player coordinates and angle
                player_body.position = game.player.pos * self.p2m
                player_body.angle = game.player.angle
                
                # Check if score has increased
                if game.score > self.score:
                    self.score = game.score
                    target_body.position = game.target * self.p2m
            space.step(dt)
        
        # Run the game
        pyglet.clock.schedule_interval(update_method, 1.0 / (game.game_config.fps * self.speedup))
        pyglet.app.run()
