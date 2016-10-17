// Author: Amit Kumar Jaiswal
// Hackerearth Handler: http://hackerearth.com/@amitkumarj441
// Twitter: @AMIT_GKP

var Promise=require('course')
var appContext={ready: false,error: false};
const INVALID_CTX = {ready: false, error: true, message: 'Course not initialized'};
function getCourseDetail(id,name,image,tags[],learners,hours,description,sing_up)
{
	return Promise.resolve('course/' + id + '.json').then(fetch);
}

function getCourse(id) { 
  // First, get a course, cache them, check an inbox, then filter course already viewed
  return getCourseDetail(id)
    .timeout(2500)                    // easy to declare expected performance, seriously awesome
    .bind(appContext)                 // subsequent Promise methods can share access to appContext obj via `this`
    .tap(course => this.course = course)    // use `.tap` to cache the result without modifying the promise result/chain
    .then(checkCourse)                 // send the returned `course` to `checkCourse` - returns an array of votes
    .tap(vote => this.vote = vote.length) // cache the # of votes for UI
    .filter(vote => !vote.viewed)       // applied to current array excludes previously viewed msgs
    .tap(unreadVotes => {
      // update UI without changing promise's return value
      if (unreadVotes.length >= 1) {
        showToast(`${unreadVotes.length} New Votes: ${unreadVotes[0].subject}...`);
      } else {
        showToast('No new votes');
      }
    })
    .tap(() => this.ready = true) // at the end of the line, set `ready` flag
    .then(checkContext) // return the app's context for further chaining
    .catch(console.error.bind(console, `Err! not very fetching: ${url}`))
}
function checkContext() { 
  if ( !this.user || !this.messages ) {
    // uh oh, override context
    return INVALID_CTX;
  }
  // otherwise ok, return ctx
  return this;
}


class CoursePromise extends Promise
{
	constructor(executor)
	{
		super((resolve,reject)=>{
		//before
		return executor(resolve,reject);
		});
		//after
}

function getCourseDetail(id,name,image,tags[],learners,hours,description,sing_up)
{
  	if (CourseCache[id])
	{
    		return Promise.resolve(CourseCache[id]);
  	}

// Use the fetch API to get the information
  	return fetch('course/' + id + '.json')
    	.then(function(result) {
      	CourseCache[id] = result;
     	return result;
})
    .catch(function()
	{
      		throw new Error('Could not find course: ' + id);
    	});
