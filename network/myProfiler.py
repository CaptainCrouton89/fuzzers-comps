import cProfile
import pipeline

cProfile.run('pipeline.main()', 'output')
