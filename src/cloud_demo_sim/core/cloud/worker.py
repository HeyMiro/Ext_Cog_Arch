#
#	Hey MiRo - CloudWorker.
#
#	Speech-to-text, chat, text-to-speech, Spotify and mp3 decoding all
#	block for hundreds of milliseconds to seconds. The HRI'25 code ran
#	them inside the 50 Hz action tick, which froze the whole robot
#	controller while it waited. Here they run on a small thread pool:
#	the tick submits a job, gets a Future back, and polls Future.done()
#	on later ticks. Exceptions stay inside the Future.
#
#	describe_error() turns an exception into a short string that is
#	safe to print: the class name plus an HTTP status if there is one.
#	We never print str(exc) of a cloud error, because some providers
#	echo (part of) the API key or the request URL in their messages.
#

import concurrent.futures
import threading



class HttpError(Exception):

	"""
	A cloud call returned a non-success HTTP status. Only the service
	name and the status code are kept: never the URL, the headers or
	the response body.
	"""

	def __init__(self, service, status_code):

		Exception.__init__(self, str(service) + " HTTP " + str(status_code))
		self.service = service
		self.status_code = status_code



def http_status(exc):

	# find an HTTP status on the usual exception shapes: our HttpError
	# and openai.APIStatusError (.status_code), requests.HTTPError
	# (.response.status_code), and a few libraries that use .status
	for obj in (exc, getattr(exc, "response", None)):
		if obj is None:
			continue
		for attr in ("status_code", "status"):
			value = getattr(obj, attr, None)
			if isinstance(value, int) and not isinstance(value, bool):
				return value
	return None


def describe_error(exc):

	# short sanitized description: class name (+ HTTP status)
	if exc is None:
		return "no error"
	text = exc.__class__.__name__
	status = http_status(exc)
	if status is not None:
		text += " " + str(status)
	return text



class CloudWorker(object):

	def __init__(self, max_workers=2, name="cloud"):

		# two threads: one conversation turn plus one side job (song
		# lookup, mp3 decode, object comment) can run at the same time
		self.name = name
		self.executor = concurrent.futures.ThreadPoolExecutor(
			max_workers=max_workers, thread_name_prefix=name)
		self.lock = threading.Lock()
		self.pending = []
		self.closed = False

	def submit(self, fn, *args, **kwargs):

		# never raises on the tick: after shutdown we hand back a
		# Future that already holds the error
		with self.lock:
			if not self.closed:
				self.pending = [f for f in self.pending if not f.done()]
				future = self.executor.submit(fn, *args, **kwargs)
				self.pending.append(future)
				return future
		future = concurrent.futures.Future()
		future.set_exception(RuntimeError(self.name + " worker is shut down"))
		return future

	def shutdown(self, wait=False):

		# cancel jobs that have not started (Python 3.8 has no
		# cancel_futures=); running jobs finish on their own, bounded
		# by the HTTP timeouts, and their results are discarded
		with self.lock:
			if self.closed:
				return
			self.closed = True
			for f in self.pending:
				f.cancel()
			self.pending = []
		self.executor.shutdown(wait=wait)
