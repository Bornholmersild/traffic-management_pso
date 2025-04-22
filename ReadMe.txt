Create random vechicles and routes
    python $SUMO_HOME/tools/randomTrips.py -n sumo/sumo_network.net.xml -r sumo/random_routes.rou.xml -b 0 -e 500 -p 2

Create grid world
    netgenerate --grid --grid.number=10 --grid.length=400 --output-file=MySUMOFile.net.xml



NOTES:
Straight line world with multiple traffic lights.
    How many times does it get stuck in a local minima
    How many iterations does it take to optimize the system

Examine which parameters and condictions changes the results.


One page
    Simple world.
    Explain parameters, world, num of cars
    Explain results.
    Explain whats the purpose.